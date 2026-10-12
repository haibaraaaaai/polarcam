"""Reconstruction of the historical intensity candidate detector.

Transcribed from the detector executed earlier in this conversation. This is
NOT a new method or a fresh re-detection. The coordinates in audit_*.npz are the
authoritative coordinates used in the delivered figures. Original image stacks
are not bundled, so agreement of this transcription with those coordinates has
not been rerun here. Supply the original stacks to perform that comparison.

Author: ClawsGO Science Agent
"""
import argparse
from pathlib import Path
import numpy as np
from scipy import ndimage


def detect_candidates(stack):
    """Return integer (y,x) coordinates and diagnostic settings.

    Input: the five observation frames in the first5 sample, no marker frame.
    The float32 mean mirrors the historical eight-file batch.
    Image processing here affects detection only, never channel extraction.
    """
    if stack.ndim != 3 or min(stack.shape[1:]) <= 20:
        raise ValueError('Expected a (frames, height, width) observation stack.')
    image = np.asarray(stack, dtype=np.float32).mean(axis=0)
    background = float(np.median(image))
    noise = float(1.4826*np.median(np.abs(image-ndimage.median_filter(image,size=5))))
    # Historical heuristic; real highly polarized cores can also trigger it.
    footprint = np.array([[1,1,1],[1,0,1],[1,1,1]])
    neighbor_max = ndimage.grey_dilation(image, footprint=footprint)
    hot = (image-neighbor_max > 8*noise) & (image > background+800)
    detection_image = np.where(hot, neighbor_max, image)
    edge = 10
    sub = detection_image[edge:-edge,edge:-edge]
    dog = (ndimage.gaussian_filter(sub,1.0,mode='nearest')-
           ndimage.gaussian_filter(sub,6.0,mode='nearest'))
    median = float(np.median(dog))
    sigma = float(1.4826*np.median(np.abs(dog-median)))
    threshold = max(0.0,median+8*sigma)
    mask = dog > threshold
    labels, number = ndimage.label(mask,structure=np.ones((3,3),int))
    small = []
    if number:
        indices = np.arange(1,number+1)
        areas = ndimage.sum(mask,labels,index=indices)
        centers = ndimage.center_of_mass(mask,labels,index=indices)
        for area,(cy,cx) in zip(areas,centers):
            if 10 <= area <= 250:
                small.append((int(cy+edge),int(cx+edge)))
    # Historical large-object rescue pass: intensity map, not a second DoG.
    smooth = ndimage.gaussian_filter(sub,8.0,mode='nearest')
    median_big = float(np.median(smooth))
    sigma_big = float(1.4826*np.median(np.abs(smooth-median_big)))
    maxima = ndimage.maximum_filter(smooth,size=21,mode='nearest')
    ys,xs = np.nonzero((smooth==maxima)&(smooth>median_big+8*sigma_big))
    large=[]
    for y,x in zip(ys,xs):
        Y,X=int(y+edge),int(x+edge)
        if all((Y-py)**2+(X-px)**2>16 for py,px in small):
            large.append((Y,X))
    # No inter-pass merging other than the >4 px check, and no extra morphology,
    # amplitude, stationarity, polarization or ROI gate at this detector stage.
    return np.asarray(small+large,dtype=int).reshape(-1,2), {
        'small':len(small),'large':len(large),'replaced_detection_pixels':int(hot.sum()),
        'dog_threshold':threshold,'image_global_median':background,'noise_proxy':noise}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stack',type=Path)
    ap.add_argument('--reference',type=Path,help='audit_*.npz to compare stored coordinates')
    args=ap.parse_args()
    spots,info=detect_candidates(np.load(args.stack,mmap_mode='r',allow_pickle=False))
    print(info)
    if args.reference:
        with np.load(args.reference,allow_pickle=False) as z: reference=z['spots']
        print('Exact ordered coordinate match:',np.array_equal(spots,reference))
        overlap=set(map(tuple,spots)) & set(map(tuple,reference))
        print('Coordinate-set overlap:',len(overlap),'of',len(reference),'stored candidates')
