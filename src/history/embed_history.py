class HistoryEmbedder:
    """
    Embeds the thought process into the history for future reference.
    """

    def __init__(self):
        # Initialize any required resources or databases
        self.history = []

    def embed(self, thought):
        """
        Embeds a thought into the history.

        Args:
            thought (str): The thought to embed.
        """
        self.history.append(thought)
        print(f"Thought embedded into history. Total thoughts: {len(self.history)}")

    def get_history(self):
        """
        Retrieves the embedded history.

        Returns:
            list: List of embedded thoughts.
        """
        return self.history