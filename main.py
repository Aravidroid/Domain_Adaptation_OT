from domain_alignment import load_data, align_domains_global_sinkhorn, align_domains_classwise_sinkhorn
from train import train_ncm, train_knn, evaluate

# Load data
Xs, ys, Xt, yt = load_data()

print("\n===== NO ADAPTATION =====")
ncm_no = train_ncm(Xs, ys)
knn1_no = train_knn(Xs, ys, k=1)
knn5_no = train_knn(Xs, ys, k=5)

print(f"NCM   → acc={evaluate(ncm_no, Xt, yt):.4f}")
print(f"kNN-1 → acc={evaluate(knn1_no, Xt, yt):.4f}")
print(f"kNN-5 → acc={evaluate(knn5_no, Xt, yt):.4f}")

print("\n===== GLOBAL OT =====")
Xs_global = align_domains_global_sinkhorn(Xs, Xt, reg=0.1)

ncm_g = train_ncm(Xs_global, ys)
knn1_g = train_knn(Xs_global, ys, k=1)
knn5_g = train_knn(Xs_global, ys, k=5)

print(f"NCM   → acc={evaluate(ncm_g, Xt, yt):.4f}")
print(f"kNN-1 → acc={evaluate(knn1_g, Xt, yt):.4f}")
print(f"kNN-5 → acc={evaluate(knn5_g, Xt, yt):.4f}")

print("\n===== CLASS-WISE OT =====")
for reg in [0.01, 0.1, 1.0]:
    Xs_cls = align_domains_classwise_sinkhorn(Xs, ys, Xt, yt, reg)

    ncm_c = train_ncm(Xs_cls, ys)
    knn1_c = train_knn(Xs_cls, ys, k=1)
    knn5_c = train_knn(Xs_cls, ys, k=5)

    print(f"\nreg={reg}")
    print(f"NCM   → acc={evaluate(ncm_c, Xt, yt):.4f}")
    print(f"kNN-1 → acc={evaluate(knn1_c, Xt, yt):.4f}")
    print(f"kNN-5 → acc={evaluate(knn5_c, Xt, yt):.4f}")
