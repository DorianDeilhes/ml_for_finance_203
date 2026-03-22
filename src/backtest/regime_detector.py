"""
regime_detector.py
------------------
Macro Regime Detection via TFT Embedding Clustering.

Extracts the learned context vectors h_t from the TFT encoder across all time
periods and applies K-Means clustering to automatically identify distinct
macroeconomic regimes (e.g., "Low Vol Growth", "High Inflation", "Crisis").

Key Features:
  1. Extract h_t embeddings for all historical observations
  2. Dimensionality reduction (optional PCA/UMAP for visualization)
  3. K-Means clustering to identify N distinct regimes
  4. Regime labeling and persistence analysis
  5. Visualization: regime timeline, feature distributions per regime

This enables:
  - Interpretable regime identification (which macro conditions cluster together?)
  - Regime-specific performance metrics (VaR accuracy per regime)
  - Regime transition analysis (how persistent are regimes?)

References:
  - Arthur & Vassilvitskii (2007): "k-means++: The advantages of careful seeding"
  - Lim et al. (2021): TFT interpretability via attention weights
"""

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects macroeconomic regimes by clustering TFT embeddings (h_t vectors).

    The TFT encoder learns to compress 63 days of macro history into a
    single vector h_t. Similar macro conditions should produce similar h_t
    vectors. By clustering these embeddings, we can discover natural groupings
    that correspond to distinct economic regimes.

    Parameters
    ----------
    n_regimes : int
        Number of regimes to detect (default: 3).
        Common choices: 2 (Bull/Bear), 3 (Low/Med/High vol), 4 (more granular).
    random_state : int
        Random seed for reproducibility.
    standardize_embeddings : bool
        Whether to standardize h_t before clustering (recommended: True).
    """

    def __init__(
        self,
        n_regimes: int = 3,
        random_state: int = 42,
        standardize_embeddings: bool = True,
    ):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.standardize_embeddings = standardize_embeddings

        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.embeddings_: Optional[np.ndarray] = None
        self.regime_labels_: Optional[np.ndarray] = None
        self.dates_: Optional[pd.DatetimeIndex] = None
        self.pca_2d_: Optional[np.ndarray] = None

    def extract_embeddings(
        self,
        model,
        data_loader,
        device: torch.device,
    ) -> np.ndarray:
        """
        Extract h_t embeddings from the TFT encoder for all sequences in data_loader.

        Parameters
        ----------
        model : ConditionalNormalizingFlow
            Trained model with a TFT encoder.
        data_loader : DataLoader
            DataLoader containing (macro_seq, returns) batches.
        device : torch.device
            Device to run inference on.

        Returns
        -------
        np.ndarray of shape (N, d_model)
            Concatenated h_t embeddings for all N observations.
        """
        model.eval()
        model.to(device)

        all_embeddings = []
        with torch.no_grad():
            for macro_seq, _ in data_loader:
                macro_seq = macro_seq.to(device)
                h_t, _ = model.tft(macro_seq)  # (batch, d_model)
                all_embeddings.append(h_t.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Extracted {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
        return embeddings

    def fit(
        self,
        embeddings: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> np.ndarray:
        """
        Fit K-Means clustering on TFT embeddings to detect regimes.

        Parameters
        ----------
        embeddings : np.ndarray of shape (N, d_model)
            TFT context vectors h_t.
        dates : pd.DatetimeIndex, optional
            Corresponding dates for each embedding (for visualization).

        Returns
        -------
        regime_labels : np.ndarray of shape (N,)
            Cluster assignment for each observation (0 to n_regimes-1).
        """
        self.embeddings_ = embeddings
        self.dates_ = dates if dates is not None else pd.date_range('2000-01-01', periods=len(embeddings))

        # Standardize embeddings before clustering
        if self.standardize_embeddings:
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)
        else:
            embeddings_scaled = embeddings

        # K-Means clustering with k-means++ initialization
        logger.info(f"Fitting K-Means with {self.n_regimes} regimes...")
        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            init='k-means++',
            n_init=20,
            max_iter=500,
            random_state=self.random_state,
        )
        self.regime_labels_ = self.kmeans.fit_predict(embeddings_scaled)

        # Compute PCA projection for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        self.pca_2d_ = pca.fit_transform(embeddings_scaled)
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        # Log regime statistics
        self._log_regime_stats()

        return self.regime_labels_

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for new embeddings.

        Parameters
        ----------
        embeddings : np.ndarray of shape (M, d_model)

        Returns
        -------
        regime_labels : np.ndarray of shape (M,)
        """
        if self.kmeans is None:
            raise RuntimeError("Must call fit() before predict()")

        if self.standardize_embeddings and self.scaler is not None:
            embeddings = self.scaler.transform(embeddings)

        return self.kmeans.predict(embeddings)

    def _log_regime_stats(self):
        """Log statistics about detected regimes."""
        logger.info("=" * 60)
        logger.info("REGIME DETECTION RESULTS")
        logger.info("=" * 60)

        for regime_id in range(self.n_regimes):
            mask = self.regime_labels_ == regime_id
            count = mask.sum()
            pct = count / len(self.regime_labels_) * 100
            logger.info(f"Regime {regime_id}: {count:4d} days ({pct:5.1f}%)")

        # Regime persistence: average consecutive days in same regime
        regime_durations = self._compute_regime_durations()
        avg_duration = np.mean([d for durations in regime_durations.values() for d in durations])
        logger.info(f"\nAverage regime persistence: {avg_duration:.1f} days")

        # Regime transition entropy (high = frequent switches)
        transitions = np.diff(self.regime_labels_)
        n_transitions = (transitions != 0).sum()
        logger.info(f"Regime transitions: {n_transitions} (entropy: {self._compute_transition_entropy():.3f})")
        logger.info("=" * 60)

    def _compute_regime_durations(self) -> Dict[int, List[int]]:
        """Compute duration of each regime episode."""
        durations = {i: [] for i in range(self.n_regimes)}
        current_regime = self.regime_labels_[0]
        current_duration = 1

        for regime in self.regime_labels_[1:]:
            if regime == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1
        durations[current_regime].append(current_duration)

        return durations

    def _compute_transition_entropy(self) -> float:
        """Compute entropy of regime transition matrix."""
        n = self.n_regimes
        transition_matrix = np.zeros((n, n))

        for i in range(len(self.regime_labels_) - 1):
            from_regime = self.regime_labels_[i]
            to_regime = self.regime_labels_[i + 1]
            transition_matrix[from_regime, to_regime] += 1

        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums

        # Compute entropy for each row, then average
        entropies = []
        for row in transition_matrix:
            row_nonzero = row[row > 0]
            if len(row_nonzero) > 0:
                entropies.append(entropy(row_nonzero))

        return float(np.mean(entropies)) if entropies else 0.0

    def get_regime_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of regime characteristics.

        Returns
        -------
        pd.DataFrame with columns:
            - regime_id
            - n_days
            - percentage
            - avg_duration
            - date_range (first to last occurrence)
        """
        if self.regime_labels_ is None:
            raise RuntimeError("Must call fit() first")

        regime_durations = self._compute_regime_durations()

        rows = []
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels_ == regime_id
            count = mask.sum()
            pct = count / len(self.regime_labels_) * 100
            avg_dur = np.mean(regime_durations[regime_id]) if regime_durations[regime_id] else 0

            regime_dates = self.dates_[mask]
            date_range = f"{pd.Timestamp(regime_dates.min()).date()} to {pd.Timestamp(regime_dates.max()).date()}"


            rows.append({
                'regime_id': regime_id,
                'n_days': count,
                'percentage': f"{pct:.1f}%",
                'avg_duration': f"{avg_dur:.1f} days",
                'date_range': date_range,
            })

        return pd.DataFrame(rows)

    def plot_regime_timeline(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 4),
    ) -> plt.Figure:
        """
        Plot regime assignments over time as a colored timeline.

        Parameters
        ----------
        output_path : str, optional
            File path to save the figure.
        figsize : tuple
            Figure dimensions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.regime_labels_ is None:
            raise RuntimeError("Must call fit() first")

        fig, ax = plt.subplots(figsize=figsize)

        # Color palette (seaborn default)
        colors = sns.color_palette("husl", self.n_regimes)

        # Plot as filled area
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels_ == regime_id
            ax.fill_between(
                self.dates_, 0, 1, where=mask,
                color=colors[regime_id], alpha=0.8,
                label=f"Regime {regime_id}",
            )

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Date", fontsize=11)
        ax.set_title("Detected Macroeconomic Regimes Over Time", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
            logger.info(f"Saved regime timeline to {output_path}")

        return fig

    def plot_regime_embedding_space(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot TFT embeddings in 2D PCA space, colored by regime.

        Shows whether regimes form tight, well-separated clusters
        (good separation = distinct macro conditions).

        Parameters
        ----------
        output_path : str, optional
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.pca_2d_ is None:
            raise RuntimeError("Must call fit() first")

        fig, ax = plt.subplots(figsize=figsize)

        colors = sns.color_palette("husl", self.n_regimes)

        for regime_id in range(self.n_regimes):
            mask = self.regime_labels_ == regime_id
            ax.scatter(
                self.pca_2d_[mask, 0],
                self.pca_2d_[mask, 1],
                c=[colors[regime_id]],
                label=f"Regime {regime_id}",
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        # Plot cluster centers (in PCA space)
        if self.kmeans is not None and self.scaler is not None:
            pca_full = PCA(n_components=2, random_state=self.random_state)
            if self.standardize_embeddings:
                pca_full.fit(self.scaler.transform(self.embeddings_))
            else:
                pca_full.fit(self.embeddings_)

            centers_pca = pca_full.transform(
                self.scaler.inverse_transform(self.kmeans.cluster_centers_)
                if self.standardize_embeddings
                else self.kmeans.cluster_centers_
            )
            ax.scatter(
                centers_pca[:, 0], centers_pca[:, 1],
                c='white', marker='X', s=300, edgecolors='black',
                linewidths=2, zorder=10, label="Centroids",
            )

        ax.set_xlabel("PCA Component 1", fontsize=11)
        ax.set_ylabel("PCA Component 2", fontsize=11)
        ax.set_title("TFT Embedding Space (PCA 2D Projection)", fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
            logger.info(f"Saved embedding space plot to {output_path}")

        return fig

    def plot_regime_transition_matrix(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 7),
    ) -> plt.Figure:
        """
        Plot regime transition probability matrix as a heatmap.

        Shows whether regimes are persistent (diagonal elements high)
        or if the model switches frequently between regimes.

        Parameters
        ----------
        output_path : str, optional
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.regime_labels_ is None:
            raise RuntimeError("Must call fit() first")

        n = self.n_regimes
        transition_matrix = np.zeros((n, n))

        for i in range(len(self.regime_labels_) - 1):
            from_regime = self.regime_labels_[i]
            to_regime = self.regime_labels_[i + 1]
            transition_matrix[from_regime, to_regime] += 1

        # Normalize rows to get transition probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_probs = transition_matrix / row_sums

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            transition_probs * 100,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Transition Probability (%)'},
            xticklabels=[f"Regime {i}" for i in range(n)],
            yticklabels=[f"Regime {i}" for i in range(n)],
            ax=ax,
        )
        ax.set_xlabel("To Regime", fontsize=11)
        ax.set_ylabel("From Regime", fontsize=11)
        ax.set_title("Regime Transition Probability Matrix", fontsize=13, fontweight="bold")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
            logger.info(f"Saved transition matrix to {output_path}")

        return fig


if __name__ == "__main__":
    # Unit test with synthetic embeddings
    np.random.seed(42)

    # Create 3 synthetic regimes with distinct embedding signatures
    regime_0 = np.random.randn(100, 64) - 1.0  # Low-vol centered at -1
    regime_1 = np.random.randn(150, 64) + 0.0  # Medium-vol centered at 0
    regime_2 = np.random.randn(80, 64) + 1.5   # High-vol centered at +1.5

    embeddings = np.vstack([regime_0, regime_1, regime_2])
    dates = pd.date_range('2020-01-01', periods=len(embeddings))

    detector = RegimeDetector(n_regimes=3, random_state=42)
    regime_labels = detector.fit(embeddings, dates=dates)

    print("\nRegime Summary:")
    print(detector.get_regime_summary())

    fig = detector.plot_regime_timeline()
    plt.show()

    fig = detector.plot_regime_embedding_space()
    plt.show()

    fig = detector.plot_regime_transition_matrix()
    plt.show()
