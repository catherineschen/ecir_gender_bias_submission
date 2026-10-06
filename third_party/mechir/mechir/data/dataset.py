from torch.utils.data import Dataset
import pandas as pd
from typing import Optional, List


class MechDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        query_field: str = "query",
        text_field: str = "text",
        perturbed_field: str = "perturbed",
        pre_perturbed: bool = False,
        additional_cols: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.pairs = pairs
        required = [query_field, text_field]
        if pre_perturbed:
            required.append(perturbed_field)
        if additional_cols:
            required.extend(additional_cols)
        for column in required:
            if column not in self.pairs.columns:
                raise ValueError(
                    f"Format not recognised, Column '{column}' not found in pairs dataframe"
                )
        self.query_field = query_field
        self.text_field = text_field
        self.perturbed_field = perturbed_field
        self.pre_perturbed = pre_perturbed
        self.additional_cols = additional_cols

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs.iloc[idx]

        # Base items
        output = [
            item[self.query_field],
            item[self.text_field],
        ]
        
        # Add perturbed field
        if self.pre_perturbed:
            output.append(item[self.perturbed_field])
            
        # --- Dynamically add all other requested columns ---
        if self.additional_cols:
            for col_name in self.additional_cols:
                output.append(item[col_name])

        return tuple(output)


__all__ = ["MechDataset"]
