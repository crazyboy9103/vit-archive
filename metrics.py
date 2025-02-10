import tensorflow as tf


class SparsePrecision(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="sparse_precision", **kwargs):
        super(SparsePrecision, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: shape (batch_size,)
        # y_pred: shape (batch_size, num_classes)
        # Convert sparse labels to one-hot vectors.
        y_true_one_hot = tf.one_hot(
            tf.cast(tf.squeeze(y_true), tf.int32), depth=self.num_classes
        )
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_one_hot = tf.one_hot(y_pred_labels, depth=self.num_classes)

        # For each class, update true positives and false positives.
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1 - y_true_one_hot) * y_pred_one_hot, axis=0)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tp * sample_weight
            fp = fp * sample_weight

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        precision_per_class = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        return tf.reduce_mean(precision_per_class)

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


class SparseRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="sparse_recall", **kwargs):
        super(SparseRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_one_hot = tf.one_hot(
            tf.cast(tf.squeeze(y_true), tf.int32), depth=self.num_classes
        )
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_one_hot = tf.one_hot(y_pred_labels, depth=self.num_classes)

        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tp * sample_weight
            fn = fn * sample_weight

        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        recall_per_class = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        return tf.reduce_mean(recall_per_class)

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


class SparseF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="f1_score", **kwargs):
        super(SparseF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", shape=(num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert sparse labels to one-hot vectors.
        y_true_one_hot = tf.one_hot(
            tf.cast(tf.squeeze(y_true), tf.int32), depth=self.num_classes
        )

        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_pred_one_hot = tf.one_hot(y_pred_labels, depth=self.num_classes)

        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tp * sample_weight
            fp = fp * sample_weight
            fn = fn * sample_weight

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )
        # F1 per class: harmonic mean of precision and recall.
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        # Macro-average F1: mean over classes.
        return tf.reduce_mean(f1)

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


class SparseAUROC(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="auc_roc", **kwargs):
        super(SparseAUROC, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.aucs = [
            tf.keras.metrics.AUC(curve="ROC", name=f"auc_{i}")
            for i in range(num_classes)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert sparse labels to one-hot vectors.
        y_true_one_hot = tf.one_hot(
            tf.cast(tf.squeeze(y_true), tf.int32), depth=self.num_classes
        )
        # Convert logits to probabilities.
        y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

        for i in range(self.num_classes):
            self.aucs[i].update_state(
                y_true_one_hot[:, i], y_pred_probs[:, i], sample_weight
            )

    def result(self):
        auc_values = [auc.result() for auc in self.aucs]
        return tf.reduce_mean(tf.stack(auc_values))

    def reset_states(self):
        for auc in self.aucs:
            auc.reset_states()
