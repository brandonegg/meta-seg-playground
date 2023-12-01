from metaseg import SegAutoMaskPredictor
import os

DIR = "/Users/brandon/Desktop"

results = SegAutoMaskPredictor().video_predict(
    source=os.path.join(DIR, "sample_webcam_feed.mov"),
    model_type="vit_l", # vit_l, vit_h, vit_b
    points_per_side=16,
    points_per_batch=64,
    min_area=1000,
    output_path=os.path.join(DIR, "output_webcam_feed.mov"),
)