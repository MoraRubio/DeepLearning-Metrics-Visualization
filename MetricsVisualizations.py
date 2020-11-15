import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix as confusion_matrix_metric

from yellowbrick.style import find_text_color
from yellowbrick.style.palettes import color_sequence
from yellowbrick.utils import div_safe

class MetricsVisualizer():
    def __init__(self, model, test_data, y_true, classes, name):
        self.model = model
        self.test_data = test_data
        y_pred = model.predict(test_data)
        if len(y_true.shape) > 1:
            self.y_true = np.argmax(y_true, axis=1)
            self.y_pred = np.argmax(y_pred, axis=1)
        else:
            self.y_true = y_true
            self.y_pred = y_pred
        self.classes = classes
        self.name = name
        self.cmap = color_sequence("YlOrRd")
        self.cmap.set_over(color="w")
        self.cmap.set_under(color="#2a7d4f")
        self._edgecolors = []
        self.fontsize = None

    def ClassificationReportViz(self, support=True):
        displayed_scores = [key for key in ("precision", "recall", "f1", "support")]

        results = precision_recall_fscore_support(self.y_true, self.y_pred)
        scores = map(lambda s: dict(zip(self.classes, s)), results)
        scores_ = dict(zip(tuple(displayed_scores), scores))

        if not support:
          displayed_scores.remove("support")
          scores_.pop("support")

        # Create display grid
        cr_display = np.zeros((len(self.classes), len(displayed_scores)))

        # For each class row, append columns for precision, recall, f1, and support
        for idx, cls in enumerate(self.classes):
            for jdx, metric in enumerate(displayed_scores):
                cr_display[idx, jdx] = scores_[metric][cls]

        # Set up the dimensions of the pcolormesh
        # NOTE: pcolormesh accepts grids that are (N+1,M+1)
        X, Y = (
            np.arange(len(self.classes) + 1),
            np.arange(len(displayed_scores) + 1),
        )

        fig, ax = plt.subplots(ncols=1, nrows=1)

        ax.set_ylim(bottom=0, top=cr_display.shape[0])
        ax.set_xlim(left=0, right=cr_display.shape[1])

        # Set data labels in the grid, enumerating over class, metric pairs
        # NOTE: X and Y are one element longer than the classification report
        # so skip the last element to label the grid correctly.
        for x in X[:-1]:
            for y in Y[:-1]:

                # Extract the value and the text label
                value = cr_display[x, y]
                svalue = "{:0.3f}".format(value)

                # Determine the grid and text colors
                base_color = self.cmap(value)
                text_color = find_text_color(base_color)

                # Add the label to the middle of the grid
                cx, cy = x + 0.5, y + 0.5
                ax.text(cy, cx, svalue, va="center", ha="center", color=text_color)

        # Draw the heatmap with colors bounded by the min and max of the grid
        # NOTE: I do not understand why this is Y, X instead of X, Y it works
        # in this order but raises an exception with the other order.
        g = ax.pcolormesh(
            Y, X, cr_display, vmin=0, vmax=1, cmap=self.cmap, edgecolor="w"
        )

        # Add the color bar
        plt.colorbar(g, ax=ax)  # TODO: Could use fig now

        # Set the title of the classifiation report
        ax.set_title("{} Classification Report".format(self.name))

        # Set the tick marks appropriately
        ax.set_xticks(np.arange(len(displayed_scores)) + 0.5)
        ax.set_yticks(np.arange(len(self.classes)) + 0.5)

        ax.set_xticklabels(displayed_scores, rotation=45)
        ax.set_yticklabels(classes)

        fig.tight_layout()

        # Return the axes being drawn on
        return ax

    def ConfusionMatrixViz(self, percent=True):
        """
        Renders the classification report; must be called after score.
        """
        labels = [0,1]
        confusion_matrix_ = confusion_matrix_metric(self.y_true, self.y_pred, labels=labels)
        class_counts_ = dict(zip(*np.unique(self.y_true, return_counts=True)))

        # Make array of only the classes actually being used.
        # Needed because sklearn confusion_matrix only returns counts for
        # selected classes but percent should be calculated on all classes
        selected_class_counts = []
        for c in labels:
            try:
                selected_class_counts.append(class_counts_[c])
            except KeyError:
                selected_class_counts.append(0)
        class_counts_ = np.array(selected_class_counts)

        # Perform display related manipulations on the confusion matrix data
        cm_display = confusion_matrix_

        # Convert confusion matrix to percent of each row, i.e. the
        # predicted as a percent of true in each class.
        if percent is True:
            # Note: div_safe function returns 0 instead of NAN.
            cm_display = div_safe(
                confusion_matrix_, class_counts_.reshape(-1, 1)
            )
            cm_display = np.round(cm_display * 100, decimals=0)

        # Y axis should be sorted top to bottom in pcolormesh
        cm_display = cm_display[::-1, ::]

        # Set up the dimensions of the pcolormesh
        n_classes = len(classes)
        X, Y = np.arange(n_classes + 1), np.arange(n_classes + 1)

        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.set_ylim(bottom=0, top=cm_display.shape[0])
        ax.set_xlim(left=0, right=cm_display.shape[1])

        # Fetch the grid labels from the classes in correct order; set ticks.
        xticklabels = classes
        yticklabels = classes[::-1]
        ticks = np.arange(n_classes) + 0.5

        ax.set(xticks=ticks, yticks=ticks)
        ax.set_xticklabels(
            xticklabels, rotation="vertical", fontsize=self.fontsize
        )
        ax.set_yticklabels(yticklabels, fontsize=self.fontsize)

        # Set data labels in the grid enumerating over all x,y class pairs.
        # NOTE: X and Y are one element longer than the confusion matrix, so
        # skip the last element in the enumeration to label grids.
        for x in X[:-1]:
            for y in Y[:-1]:

                # Extract the value and the text label
                value = cm_display[x, y]
                svalue = "{:0.0f}".format(value)
                if percent:
                    svalue += "%"

                # Determine the grid and text colors
                base_color = self.cmap(value / cm_display.max())
                text_color = find_text_color(base_color)

                # Make zero values more subtle
                if cm_display[x, y] == 0:
                    text_color = "0.75"

                # Add the label to the middle of the grid
                cx, cy = x + 0.5, y + 0.5
                ax.text(
                    cy,
                    cx,
                    svalue,
                    va="center",
                    ha="center",
                    color=text_color,
                    fontsize=self.fontsize,
                )

                # Add a dark line on the grid with the diagonal. Note that the
                # tick labels have already been reversed.
                lc = "k" if xticklabels[x] == yticklabels[y] else "w"
                self._edgecolors.append(lc)

        # Draw the heatmap with colors bounded by vmin,vmax
        vmin = 0.00001
        vmax = 99.999 if percent is True else cm_display.max()
        ax.pcolormesh(
            X,
            Y,
            cm_display,
            vmin=vmin,
            vmax=vmax,
            edgecolor=self._edgecolors,
            cmap=self.cmap,
            linewidth="0.01",
        )

        ax.set_title("{} Confusion Matrix".format(self.name))
        ax.set_ylabel("True Class")
        ax.set_xlabel("Predicted Class")

        # Call tight layout to maximize readability
        fig.tight_layout()

        # Return the axes being drawn on
        return ax