import numpy as np

class QposHistoryManager:
    """
    Maintains a history of qpos values with specified minimum length.
    If actual history is shorter than required, pads with the first qpos.
    """
    def __init__(self, config):
        self.qpos_history_min = config.get('qpos_history_min', 1)  # Default to 1 if not specified
        self.qpos_history = []
        
    def update(self, qpos):
        """
        Update history with new qpos value
        """
        # Add new qpos to history
        self.qpos_history.append(np.copy(qpos))
        
        # Optional: limit history size to prevent unbounded growth
        max_history = self.qpos_history_min + 1
        if len(self.qpos_history) > self.qpos_history_min:
            self.qpos_history = self.qpos_history[-max_history:]
    
    def get_padded_history(self):
        """
        Returns qpos history with at least qpos_history_min entries.
        If too short, pads with copies of the first value.
        """
        if not self.qpos_history:
            return []
            
        history = list(self.qpos_history)  # Create a copy to avoid modifying original
        
        # Ensure history is at least as long as required by padding with first value
        if len(history) < self.qpos_history_min:
            padding_count = self.qpos_history_min - len(history)
            padding = [np.copy(history[0])] * padding_count
            history = padding + history
            
        return history
