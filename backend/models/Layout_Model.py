# Singleton class for Layout Model from Paddle

from paddlex import create_model
from PIL import Image, UnidentifiedImageError
import os

class LayoutModel:
    _instance = None

    def __init__(self, input_dir, output_dir="./layouts", model_name="PP-DocLayout_plus-L"):
        self.output_dir = output_dir
        self.input_dir = input_dir
        os.makedirs(os.path.join(output_dir, "res"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        self.model = self.get_instance(model_name)

    @classmethod
    def get_instance(cls, model_name="PP-DocLayout_plus-L"):
        if cls._instance is None:
            cls._instance = create_model(model_name=model_name)
        return cls._instance
    
    # input: path to the one frame/slide that's currently depicted
    def run_and_store(self, frame_path):                 
        
        # error handling if frame_path does not contain image
        if not os.path.isfile(frame_path):
            raise FileNotFoundError(f"File not found: {frame_path}")

        try:
            with Image.open(frame_path) as img:
                img_width, img_height = img.size
        except UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {frame_path}")
        

        # predict output, postprocess output
        prediction = self.model.predict(
            frame_path,
            batch_size=1,
            layout_nms=True,
            threshold={10: 0.45, 12: 0.45},         # 10: doc_title, 12: header
            layout_merge_bboxes_mode="large"
        )  

        output = [prediction, img_width, img_height]
        res = self.postprocessing(output)
       
        # save results
        frame_index = os.path.splitext(os.path.basename(frame_path))[0]        

        res_json_path = os.path.join(self.output_dir, "res", f"{frame_index}.json")
        res_img_path = os.path.join(self.output_dir, "images", f"{frame_index}.png")
        res.save_to_img(save_path= res_img_path)
        res.save_to_json(save_path= res_json_path)

        return res_json_path, res_img_path
    

    def run_and_store_all_frames(self):
        if not os.path.isdir(self.input_dir):
            raise NotADirectoryError(f"Provided path is not a directory: {self.input_dir}")

        supported_extensions = {'.png', '.jpg', '.jpeg'}

        for filename in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, filename)

            if not os.path.isfile(file_path):
                continue  # Skip non-files

            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_extensions:
                continue  # Skip unsupported files

            try:
                res_json_path, res_img_path = self.run_and_store(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return self.output_dir


    def indentation_grouping(self, sorted_boxes, indent_threshold, allowed_labels):
        grouped_boxes = []
        current_group = None
        current_xmin = None  # Track minimum x of current group
        group_scores = []

        for index, box in enumerate(sorted_boxes):
            label = box['label']
            coord = box['coordinate']
            xmin, ymin, xmax, ymax = coord
            score = box['score']

            if label in ["text", "paragraph_title"]:
                if current_group is None:
                    # Start new group
                    current_group = {
                        'cls_id': 2,
                        'label': "text",
                        'coordinate': [xmin, ymin, xmax, ymax]
                    }
                    group_scores = [score]
                    current_xmin = xmin
                else:
                    if xmin > current_xmin + indent_threshold:
                        # Still part of same group (indented)
                        group_coords = current_group['coordinate']
                        group_coords[0] = min(group_coords[0], xmin)
                        group_coords[1] = min(group_coords[1], ymin)
                        group_coords[2] = max(group_coords[2], xmax)
                        group_coords[3] = max(group_coords[3], ymax)

                        group_scores.append(score)
                    else:
                        # Finalize previous group
                        current_group['score'] = min(group_scores)
                        grouped_boxes.append(current_group)

                        # Start new group
                        current_group = {
                            'cls_id': 2,
                            'label': "text",
                            'coordinate': [xmin, ymin, xmax, ymax]
                        }
                        current_xmin = xmin
                        group_scores = [score]

            elif label == "formula":
                # Always add formula to current group (if any)
                if current_group is not None:
                    group_coords = current_group['coordinate']
                    group_coords[0] = min(group_coords[0], xmin)
                    group_coords[1] = min(group_coords[1], ymin)
                    group_coords[2] = max(group_coords[2], xmax)
                    group_coords[3] = max(group_coords[3], ymax)

                    group_scores.append(score)

                # Also add the formula as a standalone box
                grouped_boxes.append(box)

            elif label in allowed_labels:
                # Finalize group if active
                if current_group is not None:
                    current_group['score'] = min(group_scores)
                    grouped_boxes.append(current_group)
                    current_group = None
                    current_xmin = None
                    group_scores = []

                # Add the unrelated box as-is
                grouped_boxes.append(box)

        # Final group at end
        if current_group is not None:
            current_group['score'] = min(group_scores)
            grouped_boxes.append(current_group)

        return grouped_boxes

    def add_IDs(self, sorted_boxes):
        for idx, box in enumerate(sorted_boxes):
            box["box_id"] = idx
        return sorted_boxes

    def postprocessing(self, output, allowed_labels=None):
        if allowed_labels is None:
            allowed_labels = ["header", "doc_title", "formula", "text", "table", "paragraph_title", "image"]

        res_dict = next(output[0])                # prediction at index 0 - generator type

        sorted_boxes = sorted(
            res_dict['boxes'],
            key=lambda box: box['coordinate'][1]  # Sort by Y-top value - top-down
        )

        img_input_width = output[1]
        res_dict['boxes'] = self.indentation_grouping(sorted_boxes, 0.025 * img_input_width, allowed_labels)      # threshold of indentation: 5% of the width of the whole slide
        res_dict['boxes'] = self.add_IDs(res_dict['boxes'])

        return res_dict
        
    