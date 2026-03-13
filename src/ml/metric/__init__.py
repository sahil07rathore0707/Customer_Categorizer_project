from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)

from src.entity.artifact_entity import ClassificationMetricArtifact


def calculate_metric(model, x, y) -> ClassificationMetricArtifact:
    """
    model: estimator
    x: input feature
    y: output feature
    """
    yhat = model.predict(x)
    
    # Calculate silhouette score if there are at least 2 clusters
    try:
        unique_labels = len(set(yhat))
        if unique_labels > 1:
            sil_score = silhouette_score(x, yhat)
        else:
            sil_score = 0.0
    except:
        sil_score = 0.0

    classification_metric = ClassificationMetricArtifact(
        f1_score=f1_score(y, yhat, average='weighted'),
        recall_score=recall_score(y, yhat, average='weighted'),
        precision_score=precision_score(y, yhat, average='weighted'),
        silhouette_score=sil_score
    )
    return classification_metric


def total_cost(y_true, y_pred):
    """
    This function takes y_ture, y_predicted, and prints Total cost due to misclassification
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = 10 * fp + 500 * fn
    return cost
