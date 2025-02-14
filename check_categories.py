from torchvision.models import resnet18, ResNet18_Weights

# Load the weights and categories
weights = ResNet18_Weights.DEFAULT
categories = weights.meta.get("categories")

# Print debug information
print("Categories loaded:", categories is not None)
print("Length:", len(categories) if categories else "N/A")
print("Sample category:", categories[0] if categories else "No category")