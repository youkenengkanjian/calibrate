import os
import urllib.request

# 创建文件夹
folder_name = "Waymo"
os.makedirs(folder_name, exist_ok=True)

# 所有要下载的文件信息（URL和重命名后的文件名）
files = [
    "Waymo_test_segment-17065833287841703_2980_000_3000_000_with_camera_labels.hdf5",
    "Waymo_test_segment-30779396576054160_1880_000_1900_000_with_camera_labels.hdf5",
    "Waymo_test_segment-89454214745557131_3160_000_3180_000_with_camera_labels.hdf5",
    "Waymo_test_segment-191862526745161106_1400_000_1420_000_with_camera_labels.hdf5",
    "Waymo_test_segment-260994483494315994_2797_545_2817_545_with_camera_labels.hdf5",
    "Waymo_test_segment-271338158136329280_2541_070_2561_070_with_camera_labels.hdf5",
    "Waymo_test_segment-272435602399417322_2884_130_2904_130_with_camera_labels.hdf5",
    "Waymo_test_segment-346889320598157350_798_187_818_187_with_camera_labels.hdf5",
    "Waymo_test_segment-366934253670232570_2229_530_2249_530_with_camera_labels.hdf5",
    "Waymo_test_segment-447576862407975570_4360_000_4380_000_with_camera_labels.hdf5",
    "Waymo_test_segment-662188686397364823_3248_800_3268_800_with_camera_labels.hdf5",
    "Waymo_test_segment-902001779062034993_2880_000_2900_000_with_camera_labels.hdf5",
    "Waymo_test_segment-933621182106051783_4160_000_4180_000_with_camera_labels.hdf5",
    "Waymo_test_segment-967082162553397800_5102_900_5122_900_with_camera_labels.hdf5",
    "Waymo_test_segment-1024360143612057520_3580_000_3600_000_with_camera_labels.hdf5",
    "Waymo_test_segment-1071392229495085036_1844_790_1864_790_with_camera_labels.hdf5",
    "Waymo_test_segment-1105338229944737854_1280_000_1300_000_with_camera_labels.hdf5",
    "Waymo_test_segment-1331771191699435763_440_000_460_000_with_camera_labels.hdf5",
    "Waymo_test_segment-1405149198253600237_160_000_180_000_with_camera_labels.hdf5",
    "Waymo_test_segment-1457696187335927618_595_027_615_027_with_camera_labels.hdf5",
    "Waymo_test_segment-1464917900451858484_1960_000_1980_000_with_camera_labels.hdf5",
    "Waymo_test_segment-1505698981571943321_1186_773_1206_773_with_camera_labels.hdf5",
    "Waymo_test_segment-1906113358876584689_1359_560_1379_560_with_camera_labels.hdf5",
    "Waymo_test_segment-1943605865180232897_680_000_700_000_with_camera_labels.hdf5",
    "Waymo_test_segment-2094681306939952000_2972_300_2992_300_with_camera_labels.hdf5",
    "Waymo_test_segment-2105808889850693535_2295_720_2315_720_with_camera_labels.hdf5",
    "Waymo_test_segment-2308204418431899833_3575_000_3595_000_with_camera_labels.hdf5",
    "Waymo_test_segment-2335854536382166371_2709_426_2729_426_with_camera_labels.hdf5",
    "Waymo_test_segment-2367305900055174138_1881_827_1901_827_with_camera_labels.hdf5",
    "Waymo_test_segment-2506799708748258165_6455_000_6475_000_with_camera_labels.hdf5",
    "Waymo_test_segment-2551868399007287341_3100_000_3120_000_with_camera_labels.hdf5",
    "Waymo_test_segment-2624187140172428292_73_000_93_000_with_camera_labels.hdf5",
    "Waymo_test_segment-2736377008667623133_2676_410_2696_410_with_camera_labels.hdf5"
]

# 下载文件
for fname in files:
    url = f"https://huggingface.co/datasets/Shengjie/WildCamera/resolve/main/dataset/Waymo/{fname}?download=true"
    output_path = os.path.join(folder_name, fname)
    print(f"Downloading {fname}...")
    urllib.request.urlretrieve(url, output_path)

print("All files downloaded successfully.")
