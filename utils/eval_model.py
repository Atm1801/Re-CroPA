import abc
import argparse
from typing import List
import torch
from PIL import Image


class BaseEvalModel(abc.ABC):
    """Base class encapsulating functionality needed to evaluate a model."""

    def __init__(self, args: List[str]):
        """Initialize model.

        Args:
            args: arguments to model. These should be parsed, or if the model
                has no applicable arguments, an error should be thrown if `args`
                is non-empty.
        """

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        """Get outputs for a batch of images and text.

        Args:
            batch_text: list of text strings, with the text "<image>" in place
                of any images to be included.
            batch_images: images to provide to model. Should be a list of lists,
              where each list contains the images for a single example.
            max_generation_length: maximum length of the generated caption.
                Defaults to 10.
            num_beams: number of beams to use for beam search. Defaults to 3.
            length_penalty: length penalty for beam search. Defaults to -2.0.

        Returns:
            List of decoded output strings.
        """
    def _prepare_images_no_normalize(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
    def get_outputs_attack(
        self,
        attack: torch.Tensor,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
        count: int,
    ) -> List[str]:
        """helper function during forward pass of attack"""
        
    def vqa_prompt(self, question, answer=None) -> str:
        """Get the prompt to use for VQA evaluation. If the answer is not provided, it should be left blank to be generated by the model.

        Returns:
            The prompt to use for VQA.
        """

    def caption_prompt(self, caption=None) -> str:
        """Get the prompt to use for caption evaluation. If the caption is not provided, it should be left blank to be generated by the model.

        Returns:
            The prompt to use for captioning.
        """

    def classification_prompt(self, class_str=None) -> str:
        """Get the prompt to use for classification evaluation. If the class_str is not provided, it should be left blank to be generated by the model.

        Returns:
            The prompt to use for classification.
        """
