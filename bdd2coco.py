import fiftyone as fo
import fiftyone.zoo as foz

source_dir = "../datasets/bdd100k"
dataset = foz.load_zoo_dataset(
    "bdd100k",
    split="validation",
    source_dir=source_dir,
    classes=["person", "rider"],
    copy_files=False,
)

# The directory to which to write the exported dataset
export_dir = "../datasets/bdd100kv3"

# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported

#Uncomment what ever format you wish to conver to

#YOLOV5
dataset_type = fo.types.YOLOv5Dataset  # for example


# Export the dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=dataset_type,
    classes=["person", "rider"],

    #export_media="copy",
    #label_field=label_field,
)