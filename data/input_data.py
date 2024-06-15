from sklearn.datasets import make_blobs, make_circles


class DatasetCreator:
    @staticmethod
    def create_blob_dataset() -> dict:
        n_samples_1 = 1000
        n_samples_2 = 100
        centers = [[0.0, 0.0], [2.0, 2.0]]
        cluster_std = [1.5, 0.5]

        X, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                          centers=centers,
                          cluster_std=cluster_std,
                          random_state=0,
                          shuffle=False)

        return {'X': X, 'y': y}

    @staticmethod
    def create_make_circles_dataset() -> dict:
        X, y = make_circles(500, factor=0.1, noise=0.1)
        return {'X': X, 'y': y}
