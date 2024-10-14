import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    
    count_parser = sub.add_parser("count")
    count_parser.add_argument("--csv", action="store_true", help="Export count results to a CSV file")
    count_parser.add_argument("--plot-all", dest="plot_all", action="store_true", help="Plot the count of objects classes in the images")
    count_parser.add_argument("--plot-cls", dest="plot_cls", action="store_true", help="Plot each object class count in every image")
    count_parser.add_argument("--save-plot", dest="save_plot", action="store_true")

    area_parser = sub.add_parser("area")
    area_parser.add_argument("--csv", action="store_true", help="Export area occupation results to a CSV file")
    area_parser.add_argument("--plot", action="store_true", help="Plot proportion of area occupied by objects in every image")
    area_parser.add_argument("--plot-cls", dest="plot_cls", action="store_true", help="Plot proportion of area occupied by objects of each class in every image")
    area_parser.add_argument("--use-boxes", dest="use_boxes", action="store_true", help="Use boxes instead of segmentation masks to make calculations")

    clf_parser = sub.add_parser("classify")
    clf_parser.add_argument("--clf-type", dest="clf_type", type="str", default="cnn_fc", help="Type of classifier to use")
    clf_parser.add_argument("--clf-ckpt", dest="clf_ckpt", type="str", default="cnn.pt", help="Path to classifier checkpoint (.skl or .pt)")
    clf_parser.add_argument("--clf-labels", dest="clf_labels", nargs="*", help="Labels for classifier predicted classes")
    clf_parser.add_argument("--clf-batch", dest="clf_batch", type=int, default=16, help="Input batch for classifier")
    clf_parser.add_argument("--show-classifications", dest="show_clf", action="store_true", help="Show image with each object classified")
    clf_parser.add_argument("--show-detections", dest="show_det", action="store_true", help="Show the original image annotated with detection boxes when showing their classifications")
    clf_parser.add_argument("--use-boxes", dest="use_boxes", action="store_true", help="Use boxes instead of segmentation masks to classify objects")

    return parser.parse_args()


def main():
    pass

if __name__ == "__main__":
    main()
