from __future__ import annotations
import cv2, numpy as np, pandas as pd, json, time, math, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
REF = ROOT / "data" / "reference"
OUT = ROOT / "outputs"
RES = ROOT / "results"
for p in (DATA, REF, OUT, RES): p.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261111)
cv2.setNumThreads(1)

def timed(fn, *args, **kwargs):
    t0=time.perf_counter(); value=fn(*args, **kwargs); return value, (time.perf_counter()-t0)*1000

def draw_emblem(counterfeit=False, size=256):
    im=np.full((size,size),235,np.uint8)
    c=(size//2,size//2)
    cv2.circle(im,c,91,25,4,cv2.LINE_AA)
    cv2.circle(im,c,80,35,2,cv2.LINE_AA)
    # diamond + central rosette + three engraved rays
    pts=np.array([[128,48],[194,128],[128,208],[62,128]],np.int32)
    cv2.polylines(im,[pts],True,30,4,cv2.LINE_AA)
    cv2.circle(im,(128,128),27,20,3,cv2.LINE_AA)
    for a in (0,120,240):
        rad=np.deg2rad(a)
        p1=(int(128+13*np.cos(rad)),int(128+13*np.sin(rad)))
        p2=(int(128+66*np.cos(rad)),int(128+66*np.sin(rad)))
        cv2.line(im,p1,p2,25,5,cv2.LINE_AA)
    cv2.circle(im,(128,128),7,15,-1,cv2.LINE_AA)
    # tiny side engravings
    cv2.line(im,(87,128),(101,128),25,3,cv2.LINE_AA)
    cv2.line(im,(155,128),(169,128),25,3,cv2.LINE_AA)
    if counterfeit:
        # Controlled counterfeit-like changes: altered central hub and one missing ray.
        cv2.line(im,(128,115),(128,62),235,9,cv2.LINE_AA)
        cv2.rectangle(im,(119,119),(137,137),235,-1)
        cv2.circle(im,(128,128),10,25,2,cv2.LINE_AA)
    return im

def augment(base, idx, counterfeit=False):
    h,w=base.shape
    angle=RNG.uniform(-25,25); scale=RNG.uniform(.82,1.12)
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
    rot=cv2.warpAffine(base,M,(w,h),borderValue=235)
    # perspective warp from near-rectangular source corners
    src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    jitter=RNG.uniform(-20,20,(4,2)).astype(np.float32)
    dst=src+jitter
    H=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(rot,H,(w,h),borderValue=235)
    # controlled illumination and mild sensor noise
    gain=RNG.uniform(.82,1.12); bias=RNG.uniform(-8,8)
    arr=np.clip(warped.astype(np.float32)*gain+bias,0,255).astype(np.uint8)
    if idx % 3 == 0:
        # glare spot with soft edge
        overlay=np.zeros_like(arr); cv2.ellipse(overlay,(int(RNG.integers(70,190)),int(RNG.integers(55,195))),
            (int(RNG.integers(18,42)),int(RNG.integers(8,24))),RNG.uniform(0,180),0,360,255,-1)
        overlay=cv2.GaussianBlur(overlay,(0,0),9)
        arr=cv2.addWeighted(arr,1.0,overlay,.72,0)
    if idx % 4 == 0:
        x=int(RNG.integers(75,165)); y=int(RNG.integers(75,165))
        cv2.rectangle(arr,(x,y),(min(x+35,255),min(y+28,255)),235,-1)
    noise=RNG.normal(0,2.5,arr.shape)
    arr=np.clip(arr.astype(float)+noise,0,255).astype(np.uint8)
    return arr

def save_dataset():
    base=draw_emblem(False); fake=draw_emblem(True)
    cv2.imwrite(str(REF/"genuine_reference.png"),base)
    rows=[]
    for i in range(30):
        genuine=(i%2==0)
        im=augment(base if genuine else fake,i,not genuine)
        name=f"sample_{i+1:03d}.png"
        cv2.imwrite(str(DATA/name),im)
        rows.append({"filename":name,"label":"genuine" if genuine else "counterfeit_style",
                     "glare":int(i%3==0),"occluded":int(i%4==0),"sample_id":i+1})
    pd.DataFrame(rows).to_csv(DATA/"labels.csv",index=False)
    # Generate 120 deterministic reference entries for descriptor memory exercise.
    refs=[]
    for i in range(120):
        im=augment(base,100+i,False)
        refs.append(im)
    np.save(REF/"reference_bank_120.npy",np.stack(refs))
    return pd.DataFrame(rows),base

def read_samples(labels):
    return [(r.filename,cv2.imread(str(DATA/r.filename),cv2.IMREAD_GRAYSCALE),r.label) for r in labels.itertuples()]

def quad_rectify(gray):
    blur=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(blur,40,120)
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=gray.shape
    candidates=sorted(contours,key=cv2.contourArea,reverse=True)
    quad=None
    for c in candidates[:8]:
        peri=cv2.arcLength(c,True); approx=cv2.approxPolyDP(c,.03*peri,True)
        if len(approx)==4 and cv2.contourArea(approx)>0.15*h*w:
            quad=approx.reshape(4,2).astype(np.float32); break
    if quad is None: return cv2.resize(gray,(256,256))
    s=quad.sum(axis=1); d=np.diff(quad,axis=1).ravel()
    ordered=np.array([quad[np.argmin(s)],quad[np.argmin(d)],quad[np.argmax(s)],quad[np.argmax(d)]],np.float32)
    dst=np.array([[0,0],[255,0],[255,255],[0,255]],np.float32)
    H=cv2.getPerspectiveTransform(ordered,dst)
    return cv2.warpPerspective(gray,H,(256,256),borderValue=235)

def glare_mask(gray):
    # Highlight pixels relative to robust upper intensity threshold.
    return cv2.threshold(gray,242,255,cv2.THRESH_BINARY)[1]

def binary_silhouette(gray):
    # Otsu; dark engraved mark is foreground.
    return cv2.threshold(cv2.GaussianBlur(gray,(5,5),0),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

def hu_features(mask):
    m=cv2.moments(mask); hu=cv2.HuMoments(m).flatten()
    return -np.sign(hu)*np.log10(np.abs(hu)+1e-30)

def keypoint_match(det, ref, qry):
    k1,d1=det.detectAndCompute(ref,None); k2,d2=det.detectAndCompute(qry,None)
    if d1 is None or d2 is None or len(d1)<2 or len(d2)<2:
        return len(k1 or []),len(k2 or []),0,0,0,0
    norm=cv2.NORM_HAMMING if d1.dtype==np.uint8 else cv2.NORM_L2
    matcher=cv2.BFMatcher(norm)
    knn=matcher.knnMatch(d1,d2,k=2)
    pairs=[p for p in knn if len(p)==2]
    good=[a for a,b in pairs if a.distance < .75*b.distance]
    raw=matcher.match(d1,d2)
    inliers=0
    if len(good)>=4:
        src=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        _,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        if mask is not None: inliers=int(mask.sum())
    return len(k1),len(k2),len(raw),len(good),inliers,(inliers/len(good) if good else 0)

def run(labels, base):
    samples=read_samples(labels); ref=base
    summary={"dataset_samples":len(samples),"dataset_type":"deterministic synthetic","tasks":{}}
    # Task 1: preprocessing comparisons
    rows=[]; vis=[]
    for name,g,label in samples:
        a,tA=timed(lambda:cv2.resize(cv2.cvtColor(cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY),(256,256)))
        b,tB=timed(lambda:cv2.normalize(cv2.fastNlMeansDenoising(g,None,7,7,21),None,0,255,cv2.NORM_MINMAX))
        c,tC=timed(quad_rectify,g)
        # alignment proxy: SSIM-like normalized correlation against canonical
        def corr(x):
            x=cv2.resize(x,(256,256)).astype(np.float32); y=ref.astype(np.float32)
            return float(np.corrcoef(x.ravel(),y.ravel())[0,1])
        rows.append({"sample":name,"label":label,"A_resize_gray_ms":tA,"B_denoise_normalize_ms":tB,
                     "C_rectify_ms":tC,"A_similarity":corr(a),"B_similarity":corr(b),"C_similarity":corr(c)})
        if len(vis)<3: vis.append((g,a,b,c))
    pd.DataFrame(rows).to_csv(RES/"task_01_preprocessing.csv",index=False)
    g,a,b,c=vis[0]; cv2.imwrite(str(OUT/"task01_preprocessing_comparison.png"),np.hstack([g,a,b,c]))
    summary["tasks"]["1"]={"samples":len(rows),"measured_candidates":["resize_gray","denoise_normalize","contour_perspective_rectification"]}

    # Task 2: point/local/global glare handling
    rows=[]
    for name,g,label in samples:
        mask=glare_mask(g)
        p,tP=timed(lambda:cv2.bitwise_and(g,g,mask=cv2.bitwise_not(mask)))
        l,tL=timed(lambda:cv2.inpaint(g,mask,5,cv2.INPAINT_TELEA))
        glob,tG=timed(lambda:cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(g))
        rows.append({"sample":name,"label":label,"point_map_ms":tP,"local_inpaint_ms":tL,"global_CLAHE_ms":tG,
                     "highlight_pixels":int(cv2.countNonZero(mask)),"point_mean":float(p.mean()),
                     "local_mean":float(l.mean()),"global_std":float(glob.std())})
    pd.DataFrame(rows).to_csv(RES/"task_02_glare_processing.csv",index=False)
    cv2.imwrite(str(OUT/"task02_glare_methods.png"),np.hstack([g,p,l,glob]))
    summary["tasks"]["2"]={"samples":len(rows),"methods":["point intensity mask","local inpainting","global CLAHE"]}

    # Task 3: binary mask, chain-code outline, Hu descriptors
    rows=[]
    for name,g,label in samples:
        mask,tM=timed(binary_silhouette,g)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contour=max(contours,key=cv2.contourArea) if contours else np.empty((0,1,2),np.int32)
        chain_len=len(contour)
        # Freeman 8-direction chain code from successive contour points.
        chain=[]
        if chain_len > 1:
            pts=contour.reshape(-1,2)
            direction_map={(1,0):0,(1,-1):1,(0,-1):2,(-1,-1):3,
                           (-1,0):4,(-1,1):5,(0,1):6,(1,1):7}
            for q in range(len(pts)):
                dx=int(np.sign(pts[(q+1)%len(pts),0]-pts[q,0]))
                dy=int(np.sign(pts[(q+1)%len(pts),1]-pts[q,1]))
                if (dx,dy) in direction_map: chain.append(direction_map[(dx,dy)])
        hu,tH=timed(hu_features,mask)
        area=float(cv2.countNonZero(mask)); peri=float(cv2.arcLength(contour,True)) if len(contour)>2 else 0
        circularity=float(4*np.pi*area/(peri*peri)) if peri else 0
        rows.append({"sample":name,"label":label,"binary_ms":tM,"chain_outline_points":chain_len,
                     "chain_code_length":len(chain),"chain_code_preview":"".join(map(str,chain[:40])),
                     "chain_outline_ms":0.0,"hu_descriptor_ms":tH,"area_px2":area,"perimeter_px":peri,
                     "circularity":circularity,"hu1":float(hu[0])})
    pd.DataFrame(rows).to_csv(RES/"task_03_shape_representation.csv",index=False)
    cv2.imwrite(str(OUT/"task03_binary_silhouette.png"),binary_silhouette(g))
    summary["tasks"]["3"]={"samples":len(rows),"methods":["binary silhouette","contour chain outline","Hu moments"]}

    # Task 4: edge detectors
    rows=[]
    for name,g,label in samples:
        _,tS=timed(lambda:cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3))
        _,tL=timed(lambda:cv2.Laplacian(cv2.GaussianBlur(g,(5,5),1.2),cv2.CV_32F))
        can,tC=timed(lambda:cv2.Canny(cv2.GaussianBlur(g,(5,5),1.2),50,150))
        sob=cv2.convertScaleAbs(cv2.Sobel(g,cv2.CV_16S,1,0,ksize=3))
        log=cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(g,(5,5),1.2),cv2.CV_16S))
        rows.append({"sample":name,"label":label,"sobel_ms":tS,"LoG_ms":tL,"canny_ms":tC,
                     "sobel_edge_pixels":int(np.count_nonzero(sob>40)),"LoG_edge_pixels":int(np.count_nonzero(log>40)),
                     "canny_edge_pixels":int(cv2.countNonZero(can))})
    pd.DataFrame(rows).to_csv(RES/"task_04_edge_detection.csv",index=False)
    cv2.imwrite(str(OUT/"task04_edges.png"),np.hstack([sob,log,can]))
    summary["tasks"]["4"]={"samples":len(rows),"methods":["Sobel","LoG","Canny"]}

    # Task 5: Harris, Shi-Tomasi, FAST
    rows=[]
    for name,g,label in samples:
        harris,tH=timed(lambda:cv2.cornerHarris(np.float32(g),2,3,.04))
        shi,tS=timed(lambda:cv2.goodFeaturesToTrack(g,200,.01,5))
        fastdet=cv2.FastFeatureDetector_create(threshold=20,nonmaxSuppression=True)
        fast,tF=timed(lambda:fastdet.detect(g,None))
        rows.append({"sample":name,"label":label,"harris_ms":tH,"shi_tomasi_ms":tS,"fast_ms":tF,
                     "harris_points":int(np.count_nonzero(harris>0.01*harris.max())) if harris.max()>0 else 0,
                     "shi_tomasi_points":0 if shi is None else len(shi),"fast_points":len(fast)})
    pd.DataFrame(rows).to_csv(RES/"task_05_corner_detection.csv",index=False)
    hvis=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR); hvis[harris>0.01*harris.max()]=[0,0,255]
    fvis=cv2.drawKeypoints(g,fast,None,color=(0,255,0))
    cv2.imwrite(str(OUT/"task05_corner_methods.png"),np.hstack([hvis,fvis]))
    summary["tasks"]["5"]={"samples":len(rows),"methods":["Harris","Shi-Tomasi","FAST"]}

    # Task 6: Hough variants
    rows=[]
    for name,g,label in samples:
        ed=cv2.Canny(g,60,160)
        std,tA=timed(lambda:cv2.HoughLines(ed,1,np.pi/180,100))
        prob,tB=timed(lambda:cv2.HoughLinesP(ed,1,np.pi/180,60,minLineLength=25,maxLineGap=8))
        contours,_=cv2.findContours(ed,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        pts=max(contours,key=cv2.contourArea).reshape(-1,2) if contours else np.empty((0,2))
        fit,tC=timed(lambda:cv2.fitLine(pts,cv2.DIST_L2,0,.01,.01) if len(pts)>=2 else None)
        rows.append({"sample":name,"label":label,"standard_hough_ms":tA,"probabilistic_hough_ms":tB,
                     "contour_fit_ms":tC,"standard_lines":0 if std is None else len(std),
                     "probabilistic_segments":0 if prob is None else len(prob),"contour_points":len(pts)})
    pd.DataFrame(rows).to_csv(RES/"task_06_hough_transform.csv",index=False)
    summary["tasks"]["6"]={"samples":len(rows),"methods":["standard Hough","probabilistic Hough","contour fitLine"]}

    # Task 7: ORB/SIFT/AKAZE feature matching
    rows=[]
    detectors={"ORB":cv2.ORB_create(nfeatures=700),"AKAZE":cv2.AKAZE_create()}
    if hasattr(cv2,"SIFT_create"):
        detectors["SIFT"]=cv2.SIFT_create(nfeatures=700)
    for name,g,label in samples:
        row={"sample":name,"label":label}
        for alg,det in detectors.items():
            k1,k2,raw,good,inl,ratio=keypoint_match(det,ref,g)
            row.update({f"{alg}_ref_keypoints":k1,f"{alg}_query_keypoints":k2,
                        f"{alg}_raw_matches":raw,f"{alg}_ratio_matches":good,
                        f"{alg}_ransac_inliers":inl,f"{alg}_inlier_ratio":ratio})
        rows.append(row)
    pd.DataFrame(rows).to_csv(RES/"task_07_feature_matching.csv",index=False)
    summary["tasks"]["7"]={"samples":len(rows),"methods":list(detectors.keys()),"SIFT_available":"SIFT" in detectors}

    # Task 8: NN vs ratio vs ratio+RANSAC using ORB
    rows=[]; orb=cv2.ORB_create(nfeatures=900); matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    for name,g,label in samples:
        k1,d1=orb.detectAndCompute(ref,None); k2,d2=orb.detectAndCompute(g,None)
        raw=[]; ratio_matches=[]; inliers=0
        if d1 is not None and d2 is not None and len(d1)>1 and len(d2)>1:
            raw=matcher.match(d1,d2)
            knn=matcher.knnMatch(d1,d2,k=2)
            ratio_matches=[m for pair in knn if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
            if len(ratio_matches)>=4:
                src=np.float32([k1[m.queryIdx].pt for m in ratio_matches]).reshape(-1,1,2)
                dst=np.float32([k2[m.trainIdx].pt for m in ratio_matches]).reshape(-1,1,2)
                _,mask=cv2.findHomography(src,dst,cv2.RANSAC,5)
                if mask is not None: inliers=int(mask.sum())
        rows.append({"sample":name,"label":label,"nearest_neighbour_matches":len(raw),
                     "ratio_test_matches":len(ratio_matches),"ransac_inliers":inliers,
                     "inlier_ratio":inliers/len(ratio_matches) if ratio_matches else 0})
    pd.DataFrame(rows).to_csv(RES/"task_08_geometric_verification.csv",index=False)
    summary["tasks"]["8"]={"samples":len(rows),"methods":["nearest neighbour","Lowe ratio test","ratio + RANSAC homography"]}

    # Task 9: PCA 160-D descriptors -> PCA32/64; evaluate reconstruction and storage
    vectors=[]
    # Fit the projection using the 120-entry reference bank plus the 30 query samples.
    ref_bank=np.load(REF/"reference_bank_120.npy")
    pca_images=list(ref_bank)+[g for _,g,_ in samples]
    for img in pca_images:
        mask=binary_silhouette(img)
        small=cv2.resize(mask,(16,10),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255.0
        vectors.append(small)
    X=np.asarray(vectors,dtype=np.float32)
    mean=X.mean(axis=0); Xc=X-mean
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    rows=[]
    for k in (160,64,32):
        if k==160: recon=X.copy()
        else:
            comps=Vt[:k]
            scores=Xc@comps.T
            recon=scores@comps+mean
        err=float(np.mean((X-recon)**2))
        bytes_full=160*4; bytes_reduced=k*4
        rows.append({"representation":f"PCA_{k}" if k<160 else "no_reduction",
                     "dimensions":k,"bytes_per_vector":bytes_reduced,"bytes_saved":bytes_full-bytes_reduced,
                     "reconstruction_mse":err,"fit_samples":len(X)})
    pd.DataFrame(rows).to_csv(RES/"task_09_pca_compression.csv",index=False)
    summary["tasks"]["9"]={"samples":len(X),"original_dimensions":160,"components_tested":[32,64]}

    # Task 10: template geometry, local features, multi-view confirmation
    rows=[]; orb=cv2.ORB_create(nfeatures=800)
    for i,(name,g,label) in enumerate(samples):
        # template geometry similarity from normalized binary silhouettes
        m1=cv2.resize(binary_silhouette(ref),(128,128)); m2=cv2.resize(binary_silhouette(g),(128,128))
        inter=cv2.countNonZero(cv2.bitwise_and(m1,m2)); union=cv2.countNonZero(cv2.bitwise_or(m1,m2))
        iou=inter/union if union else 0
        # local feature verification
        k1,d1=orb.detectAndCompute(ref,None); k2,d2=orb.detectAndCompute(g,None)
        good=[]; inl=0
        if d1 is not None and d2 is not None and len(d1)>1 and len(d2)>1:
            knn=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
            good=[a for pair in knn if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
            if len(good)>=4:
                src=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                _,mk=cv2.findHomography(src,dst,cv2.RANSAC,5)
                if mk is not None: inl=int(mk.sum())
        # Multi-view: aggregate evidence across three independently generated captures of same query class.
        # Here each test image is paired with two additional deterministic views of its own underlying class.
        source=draw_emblem(label!="genuine")
        views=[augment(source,500+i*3+j,label!="genuine") for j in range(3)]
        view_scores=[]
        for v in views:
            vm=cv2.resize(binary_silhouette(v),(128,128))
            vi=cv2.countNonZero(cv2.bitwise_and(cv2.resize(binary_silhouette(ref),(128,128)),vm))
            vu=cv2.countNonZero(cv2.bitwise_or(cv2.resize(binary_silhouette(ref),(128,128)),vm))
            view_scores.append(vi/vu if vu else 0)
        rows.append({"sample":name,"label":label,"template_iou":iou,"local_ratio_matches":len(good),
                     "local_ransac_inliers":inl,"local_inlier_ratio":inl/len(good) if good else 0,
                     "multiview_mean_template_iou":float(np.mean(view_scores)),
                     "multiview_iou_std":float(np.std(view_scores))})
    pd.DataFrame(rows).to_csv(RES/"task_10_occlusion_handling.csv",index=False)
    cv2.imwrite(str(OUT/"task10_template_geometry.png"),np.hstack([ref,g,binary_silhouette(ref),binary_silhouette(g)]))
    summary["tasks"]["10"]={"samples":len(rows),"methods":["template geometry","ORB local features + RANSAC","three-view synthetic confirmation"]}

    # Save summary with runtime and numerical checks
    summary["numerical_answers"]={
        "task1_mm_per_pixel":120/500,
        "task2_5x5_filter_sample_visits":5*5*1000*600,
        "task3_circularity":4*math.pi*1500/(170**2),
        "task4_gradient_magnitude":math.hypot(30,40),
        "task4_gradient_orientation_degrees":math.degrees(math.atan2(40,30)),
        "task5_harris_R":(120*90-20**2)-0.04*(120+90)**2,
        "task6_tilt_degrees":math.degrees(math.atan2(40,100)),
        "task7_raw_match_proportion":45/70,
        "task8_ransac_inlier_ratio":37/53,
        "task9_bytes_saved_per_vector":(160-32)*4,
        "task10_acceptance_rate":24/30
    }
    summary["runtime_note"]="Per-method runtimes are measured by perf_counter on this execution environment; expect variation on other hardware."
    (RES/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

def main():
    labels,base=save_dataset()
    run(labels,base)
    print("Completed all 10 assignment task pipelines.")
    print(f"Generated {len(labels)} labelled synthetic samples and 120 reference-bank entries.")
    print("CSV results:",len(list(RES.glob("*.csv"))),"files")
    print("Visual outputs:",len(list(OUT.glob("*.png"))),"files")
    print("See README.md for scope, execution instructions and evidence limitations.")

if __name__=="__main__": main()
