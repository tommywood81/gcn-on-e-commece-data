# ================================
# 🔧 Install Dependencies
# ================================
!pip install -q torch==2.0.0 torch-geometric scikit-learn

# ================================
# 🧩 Strategy Pattern Interfaces
# ================================
from abc import ABC, abstractmethod
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

# ================================
# 🧠 Node Strategies
# ================================
class NodeStrategy(ABC):
    @abstractmethod
    def get_node_features_and_labels(self, df: pd.DataFrame):
        pass

class TransactionNodeStrategy(NodeStrategy):
    def get_node_features_and_labels(self, df):
        features = [
            "transaction_amount", "payment_method", "product_category",
            "quantity", "customer_age", "device_used", "account_age_days", "transaction_hour"
        ]
        for col in ["payment_method", "product_category", "device_used", "customer_location"]:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        X = df[features].values
        X_scaled = StandardScaler().fit_transform(X)
        y = df["is_fraudulent"].astype(int).values
        x = torch.tensor(X_scaled, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        node_ids = df.index.tolist()
        return x, y, node_ids

# ================================
# 🔗 Edge Strategies
# ================================
class EdgeStrategy(ABC):
    @abstractmethod
    def build_edge_index(self, df: pd.DataFrame, node_ids: list) -> torch.Tensor:
        pass

class SameUserEdgeStrategy(EdgeStrategy):
    def build_edge_index(self, df, node_ids):
        edges = []
        grouped = df.groupby("user_id").groups
        for indices in grouped.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edges.append((indices[i], indices[j]))
                    edges.append((indices[j], indices[i]))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

# ================================
# 📦 Graph Builder
# ================================
def build_graph(df, node_strategy: NodeStrategy, edge_strategy: EdgeStrategy):
    x, y, node_ids = node_strategy.get_node_features_and_labels(df)
    edge_index = edge_strategy.build_edge_index(df, node_ids)
    data = Data(x=x, edge_index=edge_index, y=y)

    y_list = y.tolist()
    train_idx, test_idx = train_test_split(range(len(y_list)), test_size=0.2, stratify=y_list, random_state=42)
    data.train_mask = torch.zeros(len(y_list), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y_list), dtype=torch.bool)
    data.train_mask[torch.tensor(train_idx)] = True
    data.test_mask[torch.tensor(test_idx)] = True

    print(f"✅ Train nodes: {data.train_mask.sum().item()} | Test nodes: {data.test_mask.sum().item()}")
    return data

# ================================
# 🧠 GCN Model
# ================================
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.3, train=self.training)
        x = self.conv2(x, edge_index)
        return x

# ================================
# 🚀 Training with Weighted Loss
# ================================
def train_and_evaluate(data, epochs=10, fraud_weight=20.0):
    model = GCN(input_dim=data.num_node_features, hidden_dim=32, output_dim=2)
    class_weights = torch.tensor([1.0, fraud_weight])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        preds = out.argmax(dim=1)
        val_preds = preds[data.test_mask]
        val_labels = data.y[data.test_mask]
        val_acc = accuracy_score(val_labels.tolist(), val_preds.tolist())
        print(f"Epoch {epoch}: Loss={loss.item():.4f} | Val Acc={val_acc:.4f}")

    print("\n📊 Final Evaluation:")
    if val_preds.numel() > 0 and val_labels.numel() > 0:
        print(classification_report(val_labels.tolist(), val_preds.tolist(), target_names=["Not Fraud", "Fraud"]))
    else:
        print("⚠️ No test predictions available. Check your test mask or data split.")

# ================================
# 🧪 Run Pipeline
# ================================
def run_pipeline(node_strategy: NodeStrategy, edge_strategy: EdgeStrategy, fraud_weight=20.0):
    print(f"\nRunning with node strategy: {node_strategy.__class__.__name__}")
    print(f"Running with edge strategy: {edge_strategy.__class__.__name__}")
    print(f"Using fraud class weight: {fraud_weight}")

    url = "https://raw.githubusercontent.com/tommywood81/GCN_Data/refs/heads/data/Fraudulent_E-Commerce_Transaction_Data_2.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["user_id"] = df["customer_location"]      # simulate user ID

    data = build_graph(df, node_strategy, edge_strategy)
    train_and_evaluate(data, epochs=10, fraud_weight=fraud_weight)

# ================================
# ▶️ Run the Script
# ================================
run_pipeline(
    node_strategy=TransactionNodeStrategy(),
    edge_strategy=SameUserEdgeStrategy(),
    fraud_weight=10.0  # try 10.0–50.0 depending on performance
)
