import argparse
import sys
import cv2
import numpy as np

from core.filter import GuidedFilter
from core.applications import detail_enhancement, dehaze, joint_filtering
from cv.image import to_32F, to_8U


def main():
    parser = argparse.ArgumentParser(
        description="Guided Image Filter - A Python implementation of He et al.'s Guided Image Filtering (TPAMI '12)"
    )
    
    # Input/Output options
    parser.add_argument("-i", "--input", required=True, help="Path to input image to be filtered")
    parser.add_argument("-o", "--output", default="output.png", help="Path to save the output image")
    parser.add_argument("-g", "--guide", help="Path to guide image (default: input image is used as guide)")
    
    # Task selection
    parser.add_argument(
        "-t", "--task", 
        choices=["smooth", "enhance", "dehaze", "joint"], 
        default="smooth", 
        help="Task to perform: 'smooth' (edge-preserving smoothing), 'enhance' (detail enhancement), 'dehaze' (haze removal), 'joint' (joint filtering using separate guide)"
    )
    
    # Parameters for Guided Filter
    parser.add_argument("-r", "--radius", type=int, default=4, help="Filter radius r (kernel size is 2*r+1)")
    parser.add_argument("-e", "--eps", type=float, default=0.01, help="Regularization parameter epsilon (controls edge preservation)")
    
    # Parameters for Detail Enhancement
    parser.add_argument("-f", "--factor", type=float, default=3.0, help="Detail amplification factor for enhancement task")
    parser.add_argument("--no-color-preserve", action="store_true", help="Disable YCrCb luminance-only detail enhancement")
    
    # Parameters for Dehazing
    parser.add_argument("--omega", type=float, default=0.95, help="Haze preservation factor (0.0 to 1.0) for dehazing")
    parser.add_argument("--t0", type=float, default=0.1, help="Lower bound for transmission map in dehazing")
    parser.add_argument("--guided-radius", type=int, default=40, help="Guided filter radius for transmission map refinement")
    parser.add_argument("--guided-eps", type=float, default=0.001, help="Guided filter epsilon for transmission map refinement")
    
    args = parser.parse_args()
    
    # 1. Load input image
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        print(f"Error: Could not load input image from '{args.input}'", file=sys.stderr)
        sys.exit(1)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Process based on task
    print(f"Running task '{args.task}' on '{args.input}'...")
    
    if args.task == "smooth":
        # Guide image defaults to the input image itself
        if args.guide:
            guide_bgr = cv2.imread(args.guide)
            if guide_bgr is None:
                print(f"Error: Could not load guide image from '{args.guide}'", file=sys.stderr)
                sys.exit(1)
            guide_rgb = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2RGB)
        else:
            guide_rgb = img_rgb
            
        gf = GuidedFilter(guide_rgb, radius=args.radius, eps=args.eps)
        out_rgb = gf.filter(img_rgb)
        
    elif args.task == "enhance":
        out_rgb = detail_enhancement(
            img_rgb, 
            radius=args.radius, 
            eps=args.eps, 
            factor=args.factor, 
            color_space_preserve=not args.no_color_preserve
        )
        
    elif args.task == "dehaze":
        out_rgb = dehaze(
            img_rgb,
            radius=15,
            eps=args.eps,
            omega=args.omega,
            t0=args.t0,
            guided_radius=args.guided_radius,
            guided_eps=args.guided_eps
        )
        
    elif args.task == "joint":
        if not args.guide:
            print("Error: The 'joint' task requires a separate guide image (--guide / -g)", file=sys.stderr)
            sys.exit(1)
        guide_bgr = cv2.imread(args.guide)
        if guide_bgr is None:
            print(f"Error: Could not load guide image from '{args.guide}'", file=sys.stderr)
            sys.exit(1)
        guide_rgb = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2RGB)
        
        out_rgb = joint_filtering(img_rgb, guide_rgb, radius=args.radius, eps=args.eps)
        
    else:
        print(f"Error: Unknown task '{args.task}'", file=sys.stderr)
        sys.exit(1)
        
    # 3. Save output image
    out_bgr = cv2.cvtColor(to_8U(out_rgb), cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(args.output, out_bgr)
    if success:
        print(f"Successfully processed image. Saved result to '{args.output}'")
    else:
        print(f"Error: Could not write output image to '{args.output}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
