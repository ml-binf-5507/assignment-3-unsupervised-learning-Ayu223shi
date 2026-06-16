import inspect
import pkgutil
import students.dimensionality_reduction as dr
import students.clustering as cl
import students.visualization as vz


REQUIRED = [
    dr.apply_pca,
    dr.apply_tsne,
    dr.apply_umap,
    cl.run_kmeans,
    cl.run_hierarchical,
    vz.plot_2d,
    vz.plot_representation_comparison,
    vz.plot_cluster_comparison,
]


def main():
    missing = []
    for func in REQUIRED:
        if 'NotImplementedError' in inspect.getsource(func):
            missing.append(func.__name__)
    if missing:
        print("The following functions are still unimplemented:", missing)
        exit(1)
    else:
        print("All required functions appear implemented.")
        exit(0)


if __name__ == "__main__":
    main()
