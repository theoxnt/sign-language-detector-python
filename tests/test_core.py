import sys
from unittest.mock import MagicMock, call, patch
sys.path.append('..')
from src.core import *
import unittest
import numpy as np
from pathlib import Path


class TestCore(unittest.TestCase):

    @patch('src.core.cv2.VideoCapture')
    @patch('src.core.cv2.imshow')
    @patch('src.core.cv2.waitKey')
    @patch('src.core.cv2.imwrite')
    @patch('src.core.cv2.destroyAllWindows')
    @patch('src.core.os.makedirs')
    @patch('src.core.os.path.exists')
    def test_collect_images(
        self, 
        mock_exists, 
        mock_makedirs,
        mock_destroyAllWindows,
        mock_imwrite, 
        mock_waitKey, 
        mock_imshow, 
        mock_VideoCapture
        ):

    
        mock_VideoCapture.return_value.isOpened.return_value = True

        mock_exists.return_value = False

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_VideoCapture.return_value.read.return_value = (True, fake_frame)

        mock_waitKey.return_value = ord('q')

        num_classes = 2
        images_per_class = 3
        folder_name = "test_folder"
        result = collect_images(num_classes, images_per_class, folder_name)

        self.assertTrue(result)

        self.assertEqual(mock_imwrite.call_count, num_classes * images_per_class)

        self.assertEqual(mock_makedirs.call_count, num_classes + 1) # +1 for main folder
        expected_calls = [
            call(f'./src/data/{folder_name}'),
            call(f'./src/data/{folder_name}/0'),
            call(f'./src/data/{folder_name}/1')
        ]
        mock_makedirs.assert_has_calls(expected_calls, any_order=False)

        self.assertEqual(mock_VideoCapture.return_value.isOpened.call_count, 1)
        self.assertEqual(mock_VideoCapture.return_value.release.call_count, 1)

        self.assertEqual(mock_destroyAllWindows.call_count, 1)
    

    @patch("mediapipe.solutions.hands.Hands.process")
    @patch("os.listdir")
    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open")
    @patch("pickle.dump")
    def test_create_dataset(
        self,
        mock_pickle,
        mock_open,
        mock_mkdir,
        mock_cvtColor,
        mock_imread,
        mock_listdir,
        mock_hands_process
    ):
        
        fake_hand_landmarks = MagicMock()
        fake_hand_landmarks.landmark = [MagicMock(x=0.1*i, y=0.05*i) for i in range(21)]
        mock_hands_process.return_value = MagicMock(
            multi_hand_landmarks=[fake_hand_landmarks]
        )

        def listdir_side_effect(path):
            if path == "./src/data":
                return ["test_folder_1", "test_folder_2"]
            elif "0" in path:
                return ["0.jpg", "1.jpg", "2.jpg"]
            elif "1" in path:
                return ["0.jpg", "1.jpg", "2.jpg"]
            return []
        mock_listdir.side_effect = listdir_side_effect

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = fake_frame

        result = create_dataset(number_of_classes=2, dataset_name="dataset_test")

        self.assertTrue(result)

        mock_mkdir.assert_called_once() 

        mock_open.assert_called_once()

        mock_pickle.assert_called_once() 

        self.assertEqual(mock_listdir.call_count, 6)

    
    def test_train_classsifier(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)

    def test_train_forest(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)

    def test_train_neural_network(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)
    
    def test_epoch_trainer(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)

    def test_splitting_dataset(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)
    
    def test_inference_classifier(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()