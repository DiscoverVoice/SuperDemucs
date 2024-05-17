from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 예측
model.eval()
val_predictions = []
val_true_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        val_predictions.extend(preds.numpy())
        val_true_labels.extend(labels.numpy())

# 평가 보고서
print(classification_report(val_true_labels, val_predictions, target_names=label_encoder.classes_))

# 혼동 행렬 시각화
cm = confusion_matrix(val_true_labels, val_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
