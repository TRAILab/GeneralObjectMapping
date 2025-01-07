"""

A virtual dataset to be used.

"""

import numpy as np


class Frame:
    """
    A frame structure
    """

    def __init__(self):
        self.rgb = None
        self.depth = None
        self.mask = None


class GeneralDataset:
    """
    A virtual template for any dataset.
    """

    def __init__(self, data_root):

        # Init basic information
        self.data_root = data_root

        self.scene_names = []

    def load_subset_scene_name_list(self, subset="all"):
        """
        There is an option to only consider a subset of this dataset.

        Return: A subset of scene_names
        """
        if subset == "all":
            return self.scene_names
        else:
            raise NotImplementedError

    def get_scene_name_list(self):
        """
        Get the list of scene names in the dataset.

        Returns:
            list: A list of scene names.
        """
        return self.scene_names

    def load_scene_list(self, subset="all"):
        """
        Input:
            - subset: 'all' or 'train' or 'test' [optional]

        Output:
            scene_names
            scene_info

        """

        pass

    def load_objects_orders_from_scene_with_category(self, scene_name, category=None):
        """
        Load the object orders from a scene with a specific category.

        Args:
            scene_name (str): The name of the scene.
            category (str): The category of the objects.

        Returns:
            list: A list of object orders.
        """

        pass

    def get_scene_num(self):
        """
        Get the number of scenes in the dataset.

        Returns:
            int: The number of scenes.
        """
        return len(self.scene_names)

    def get_frame_num(self, scene_name):
        """
        Get the number of frames in a scene.

        Args:
            scene_name (str): The name of the scene.

        Returns:
            int: The number of frames.
        """
        pass

    def get_frames_by_scene(self, scene_name, N=None):
        """
        Load a frame structure, including RGB images, Depth images etc.
        """

        num_frames = self.get_frame_num(scene_name)

        if N is None:
            N = num_frames

        if N > num_frames:
            raise ValueError(
                "The number of frames you want to load is larger than the number of frames in the scene"
            )

        # Sample N frames
        frame_idx = np.random.choice(num_frames, N, replace=False)

        # Load frame
        frame_list = []
        for idx in frame_idx:
            frame = self.load_frame(scene_name, idx)
            frame_list.append(frame)

        return frame_list

    def load_frame(self, scene_name, frame_id):
        """
        Load a frame structure, including RGB images, Depth images etc.
        """
        pass
