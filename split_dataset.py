import argparse, os, shutil, random
from pathlib import Path

def is_img(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def copy_subset(files, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Thư mục gốc dạng Kaggle của bạn, VD: .../vietnames banknotes/dataset")
    ap.add_argument("--out", default="data", help="Thư mục đích: tạo train/val/test bên trong (mặc định: data)")
    ap.add_argument("--train", type=float, default=0.7, help="tỉ lệ train (default 0.7)")
    ap.add_argument("--val", type=float, default=0.15, help="tỉ lệ val (default 0.15)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    random.seed(args.seed)

    assert abs(args.train + args.val - 1.0) < 1e-6 or args.train + args.val < 1.0, \
        "train+val phải <= 1.0 (phần còn lại sẽ là test)"

    test_ratio = max(0.0, 1.0 - (args.train + args.val))
    print(f"Split ratios: train={args.train:.2f}, val={args.val:.2f}, test={test_ratio:.2f}")

    classes = [d.name for d in src.iterdir() if d.is_dir()]
    assert len(classes) > 1, "Không thấy thư mục lớp nào trong src."
    classes.sort()
    print("Classes:", classes)

    # Xoá out nếu muốn làm sạch (an toàn: chỉ xoá các nhánh con)
    for sub in ["train", "val", "test"]:
        p = out / sub
        if p.exists():
            shutil.rmtree(p)

    total = 0
    for cls in classes:
        cls_dir = src / cls
        imgs = [p for p in cls_dir.iterdir() if p.is_file() and is_img(p)]
        if not imgs:
            print(f"⚠️  Bỏ qua lớp {cls} (không có ảnh).")
            continue
        imgs.sort()
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * args.train)
        n_val   = int(n * args.val)
        n_test  = n - n_train - n_val

        train_files = imgs[:n_train]
        val_files   = imgs[n_train:n_train+n_val]
        test_files  = imgs[n_train+n_val:]

        copy_subset(train_files, out / "train" / cls)
        copy_subset(val_files,   out / "val" / cls)
        copy_subset(test_files,  out / "test" / cls)

        total += n
        print(f"{cls:>8}: {n:4d} → train {len(train_files):4d}, val {len(val_files):4d}, test {len(test_files):4d}")

    print(f"Done. Tổng ảnh: {total}. Output tại: {out.resolve()}")

if __name__ == "__main__":
    main()
