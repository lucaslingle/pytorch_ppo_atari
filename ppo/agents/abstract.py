from abc import ABC, abstractmethod
import torch as tc


class AgentModel(ABC, tc.nn.Module):
    @abstractmethod
    def forward(self, x):
        """
        Forward.
        """
        pass