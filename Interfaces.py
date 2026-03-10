from abc import abstractmethod, ABC
from torch import nn, Tensor
from torch_geometric.data import Data

class LinkPredictor(ABC, nn.Module):
	
	@abstractmethod
	def forward(
		edge_index: Tensor,
		edge_attr: Tensor | None = None,
		edge_mask: Tensor | None = None,
		node_mask: Tensor | None = None,
		edge_weight: Tensor | None = None,
		node_weight: Tensor | None = None, 
		**kwargs
	) -> tuple[Tensor, Tensor]:
		"""
		Returns: 
			y_logits (Tensor): Edge predictions  
			embeddings (Tensor): Edge embeddings 
		"""
		pass


class Explainer(ABC, nn.Module):
	
	@abstractmethod
	def __init__(self, model: LinkPredictor, **kwargs):
		super().__init__(**kwargs)
		self.model = model

	@abstractmethod
	def explain(self, **kwargs):
		pass


class ParametricExplainer(ABC, Explainer):

	@abstractmethod
	def fit(self, G: Data, **kwargs):
		pass
