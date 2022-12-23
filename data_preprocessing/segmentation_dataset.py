import os, shutil, sys
# 원본 데이터셋을 압축 해제한 디렉터리 경로
original_dataset_dir = '../CNN_image/categorical/train'

# 소규모 데이터셋을 저장할 디렉터리
base_dir = '../CNN_image/train_small'
if os.path.exists(base_dir):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
    shutil.rmtree(base_dir)  # 이 코드는 책에 포함되어 있지 않습니다.
os.mkdir(base_dir)
# 훈련, 검증, 테스트 분할을 위한 디렉터리
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)
# 훈련용 디렉터리
train_field_dir = os.path.join(train_dir, 'field')
os.mkdir(train_field_dir)
train_island_dir = os.path.join(train_dir, 'island')
os.mkdir(train_island_dir)
train_mountain_dir = os.path.join(train_dir, 'mountain')
os.mkdir(train_mountain_dir)
train_urban_dir = os.path.join(train_dir, 'urban')
os.mkdir(train_urban_dir)

# 검증용 rural 사진 디렉터리
validation_field_dir = os.path.join(validation_dir, 'field')
os.mkdir(validation_field_dir)
validation_island_dir = os.path.join(validation_dir, 'island')
os.mkdir(validation_island_dir)
validation_mountain_dir = os.path.join(validation_dir, 'mountain')
os.mkdir(validation_mountain_dir)
validation_urban_dir = os.path.join(validation_dir, 'urban')
os.mkdir(validation_urban_dir)
import random

fnames_fi = ['field{}.png'.format(i) for i in range(1, 424)]
train_fi_fnames = random.sample(fnames_fi, 400)
fnames_is = ['island{}.png'.format(i) for i in range(1, 410)]
train_is_fnames = random.sample(fnames_is, 400)
fnames_mo = ['mountain{}.png'.format(i) for i in range(1, 418)]
train_mo_fnames = random.sample(fnames_mo, 400)
fnames_ur = ['urban{}.png'.format(i) for i in range(1, 452)]
train_ur_fnames = random.sample(fnames_ur, 400)
val_fi = random.sample(fnames_fi, 50)
val_is = random.sample(fnames_is, 50)
val_mo = random.sample(fnames_mo, 50)
val_ur = random.sample(fnames_ur, 50)
# len(train_ru_fnames)
# 각 카테고리별 이미지를 train에 복사합니다
for fname in train_fi_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_field_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_fi:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_field_dir, fname)
    shutil.copyfile(src, dst)

for fname in train_is_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_island_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_is:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_island_dir, fname)
    shutil.copyfile(src, dst)

for fname in train_mo_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_mountain_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_mo:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_mountain_dir, fname)
    shutil.copyfile(src, dst)

for fname in train_ur_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_urban_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_ur:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_urban_dir, fname)
    shutil.copyfile(src, dst)
# # 다음 500개 강아지 이미지를 test_dogs_dir에 복사합니다
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)
print('훈련용 rural 이미지 전체 개수:', len(os.listdir(train_mountain_dir)))
훈련용
rural
이미지
전체
개수: 400