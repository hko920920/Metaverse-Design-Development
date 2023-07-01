import shutil

bvh_file = "avatar.bvh"

src = f"C:/Users/woduc/Desktop/inde_unity/gesticulator-master/demo/{bvh_file}"
dst = "C:/Users/woduc/Desktop/inde_unity/Assets"
shutil.copy(src, dst)