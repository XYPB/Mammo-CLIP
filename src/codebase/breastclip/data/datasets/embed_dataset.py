import os
import pickle
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from breastclip.data.data_utils import load_transform
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img


def get_imgs(img_path, scale, transform=None, multiscale=False):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x, scale)
    img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(img)

    return img

# #############################################
# EMBED constants
# #############################################
DATA_BASE_DIR = '~/project/PEMedCLIP/data' 
DATA_BASE_DIR = os.path.expanduser(DATA_BASE_DIR)
EMBED_DATA_DIR = DATA_BASE_DIR + "/Embed"
EMBED_DATA_PATH = EMBED_DATA_DIR + "/images"
EMBED_TRAIN_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_train.csv"
EMBED_TEST_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_test.csv"
EMBED_VALID_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_valid.csv"
# Read the full annotation for calcification information
EMBED_ANNO_CSV_REDUCED = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical_reduced.csv"
EMBED_ANNO_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical.csv"
EMBED_LEGENDS_CSV = EMBED_DATA_DIR + "/tables/AWS_Open_Data_Clinical_Legend.csv"
EMBED_INTER_VIEW_MAP = EMBED_DATA_DIR + "/tables/img_path2inter_view.pkl"
EMBED_INTER_SIDE_MAP = EMBED_DATA_DIR + "/tables/img_path2inter_side.pkl"
EMBED_BALANCED_TEST_PATH = EMBED_DATA_DIR + "/test_7x200_path2label.pickle"
EMBED_10PCT_TEST_PATH = EMBED_DATA_DIR + "/test_10pct_path2label.pickle"
EMBED_BALANCED_TRAIN_PATH = EMBED_DATA_DIR + "/train_7x550_path2label.pickle"
EMBED_BALANCED_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_4x500_path2density.pickle"
EMBED_10PCT_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_10pct_path2density.pickle"
EMBED_BALANCED_LARGE_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_4x2500_path2density.pickle"
EMBED_BALANCED_DEN_TRAIN_PATH = EMBED_DATA_DIR + "/train_4x1000_path2density.pickle"
EMBED_TRAIN_PATH2DENSITY = EMBED_DATA_DIR + "/train_path2density.pickle"
EMBED_VALID_PATH2DENSITY = EMBED_DATA_DIR + "/valid_path2density.pickle"
EMBED_TEST_PATH2DENSITY = EMBED_DATA_DIR + "/test_path2density.pickle"
EMBED_TRAIN_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_train_resized.pickle"
EMBED_VALID_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_valid_resized.pickle"
EMBED_TEST_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_test_resized.pickle"

EMBED_IMAGE_TYPE_COL = "FinalImageType"
EMBED_PATH_COL = "anon_dicom_path"
EMBED_PID_COL = 'empi_anon'
EMBED_SID_COL = 'acc_anon'
EMBED_SIDE_COL = 'ImageLateralityFinal'
EMBED_FINDING_SIDE_COL = 'side'
EMBED_VIEW_COL = 'ViewPosition'
EMBED_DENSITY_COL = 'tissueden'
EMBED_BIRADS_COL = 'asses'
EMBED_PROCEDURE_COL = 'StudyDescription'
EMBED_MASS_SHAPE_COL = 'massshape'
EMBED_MASS_DENSITY_COL = 'massdens'
EMBED_CALC_FIND_COL = 'calcfind'
EMBED_CALC_DIST_COL = 'calcdistri'
EMBED_AGE_COL = 'age_at_study'
EMBED_ROI_COORD = 'ROI_coords'
EMBED_RACE_COL = 'RACE_DESC'
EMBED_ETHNIC_COL = 'ETHNIC_GROUP_DESC'
EMBED_PATH_TRANS_FUNC = lambda x: x.replace("/mnt/NAS2/mammo/anon_dicom", EMBED_DATA_PATH)
EMBED_PROCEDURE2REASON_FUNC = lambda x: "screening" if "screen" in x.lower() else "diagnostic" if "diag" in x.lower() else ""
# Normal caption constants
BREAST_BASE_CAPTION = "This is a breast 2D full-field digital mammogram of a patient "
BREAST_SIDE_CAPTION = "on side " # Make the caption more grammarly correct
BREAST_VIEW_CAPTION = "with view "
BREAST_DENSITY_CAPTION = "with breast tissue density "
BREAST_BIRADS_CAPTION = "with BIRADS score "
# TODO: Add more findings according to the EMBED dataset structure
# Natural Captions
EMBED_NATURE_BASE_CAPTION = "This is a breast 2D full-field digital {{REASON}} mammogram of a patient. "
EMBED_NATURE_IMAGE_CAPTION = "This mammogram is for {{SIDE}} breast with {{VIEW}} view. "
# Structural Captions
EMBED_PROCEDURE = 'Procedure reported: ' # EMBED_PROCEDURE_COL
EMBED_REASON = 'Reason for procedure: ' # Screening / Diagnostic, maybe add more details later
EMBED_PATIENT = 'Patient info: ' # AGE + RACE + ETHNIC
EMBED_IMAGE = 'Image info: ' # EMBED_IMAGE_TYPE_COL + EMBED_SIDE_COL + EMBED_VIEW_COL
EMBED_DENSITY = 'Breast composition: ' # EMBED_DENSITY_COL + extra description
EMBED_FINDINGS = 'Findings: ' # EMBED_MASS info + EMBED_CALC_FIND_COL + extra description
EMBED_IMPRESSIONS = 'Impressions: ' # EMBED_BIRADS_COL + extra description
EMBED_ASSESSMENT = 'Overall Assessment: ' # EMBED_BIRADS_COL number

EMBED_PATIENT_INFO_CAPTION = "This patient is {{RACE}}, {{ETHNIC}}, and {{AGE}} years old. "
EMBED_IMAGE_INFO_CAPTION = "This is a {{IMAGE_TYPE}} full-field digital mammogram of the {{SIDE}} breast with {{VIEW}} view. "
EMBED_BREAST_COMPOSITION_CAPTION = "The breast is {{DENSITY}}. "
EMBED_DENSITY_EXTRA_CAPTION = {
    3: "This may lower the sensitivity of mammography. ",
    4: "This may lower the sensitivity of mammography. ",
}
EMBED_FINDS_CAPTION = "The mammogram shows that "
EMBED_MASS_CAPTION = {
    'A': "an additional imaging is recommended. ",
    'N': "no significant masses, calcification, or other abnormalities are present. ",
    'B': "a benign finding is present. ",
    'P': "a probably benign finding is present. ",
    'S': "a suspicious abnormality is present. ",
    'M': "a highly suggestive of malignancy is present, a biopsy is recommended. ",
    'K': "a known biopsy-proven malignant mass is present. ",
}
EMBED_MASS_EXTRA_CAPTION = 'The mass is {{SHAPE}} and {{DENSITY}}. '
EMBED_CALC_FINDS_CAPTION = 'A {{DISTRI}} {{SHAPE}} calcification is present. '
EMBED_IMPRESSION_CAPTION = "BI-RADS Category {{BIRADS}}: {{BIRADS_DESC}}. "
EMBED_ASSESSMENT_CAPTION = {
    'A': "Additional imaging is recommended. ",
    'N': "Negative. ",
    'B': "Benign. ",
    'P': "Probably benign. ",
    'S': "Suspicious abnormality. ",
    'M': "Highly suggestive of malignancy. ",
    'K': "Known biopsy-proven malignancy. ",
}
EMBED_SIDES_DESC = {
    'L': 'left',
    'R': 'right',
    'B': 'bilateral',
}
EMBED_DENSITY_DESC = {
    1: "almost entirely fat",
    2: "scattered fibroglandular densities",
    3: "heterogeneously dense",
    4: "extremely dense",
    5: "normal male dense",
}
EMBED_LETTER_TO_BIRADS = {
    "A": 0,
    "N": 1,
    "B": 2,
    "P": 3,
    "S": 4,
    "M": 5,
    "K": 6,
}
EMBED_BIRADS_DESC = {
    'A': "additional imaging required",
    'N': "negative",
    'B': "benign finding",
    'P': "probably benign finding",
    'S': "suspicious abnormality",
    'M': "highly suggestive of malignancy",
    'K': "known biopsy-proven malignancy",
}
GET_JPEG_PATH_FUNC = lambda x: x.replace('Embed', 'EMBED_1080_JPG').replace(".dcm", "_resized.jpg")
GET_ALIGNED_MLO_FUNC = lambda x: x.replace(".jpg", "_align_to_cc.jpg")

def check_element_type(element, str_pool=None):
    if str_pool is None:
        # either non-empty string or non-nan float
        return (isinstance(element, str) and element != "") or (
            isinstance(element, float) and not np.isnan(element)
        )
    else:
        # either string in pool or non-nan float
        return (isinstance(element, str) and element in str_pool) or (
            isinstance(element, float) and not np.isnan(element)
        )

class EmbedPretrainingDataset(data.Dataset):
    def __init__(
        self,
        tokenizer,
        split="train",
        dataset="embed",
        transform_config=None,
        data_pct=1.0,
        imsize=1024,
        text_max_length=256,
        mean=0,
        std=0,
        simple_cap=False,
        train_sub_set=False,
        structural_cap=True,
        natural_cap=False,
        balanced_test=True,
        pred_density=False,
        ten_pct=True,
        large_density=False,
        instance_test_cap=False,
        screen_only=True, 
        aligned_mlo=False,
        zero_shot=False,
        **kwargs,
    ):
        super().__init__()
        if not os.path.exists(EMBED_DATA_DIR):
            raise RuntimeError(f"{EMBED_DATA_DIR} does not exist!")

        self.transform = load_transform(split=split, transform_config=transform_config)
        self.imsize = imsize
        split = "test"
        self.split = "test"
        self.dataset = dataset
        self.text_max_length = text_max_length
        self.structural_cap = structural_cap
        self.simple_cap = simple_cap
        self.natural_cap = natural_cap
        self.balanced_test = balanced_test
        self.pred_density = pred_density
        self.instance_test_cap = instance_test_cap
        self.zero_shot = zero_shot
        self.screen_only = screen_only
        self.aligned_mlo = aligned_mlo
        self.mean = mean
        self.std = std
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        if split == "train":
            self.df = pd.read_csv(EMBED_TRAIN_META_CSV)
        elif split == "valid":
            self.df = pd.read_csv(EMBED_VALID_META_CSV)
        elif split == "test":
            self.df = pd.read_csv(EMBED_TEST_META_CSV)
            self.cls_prompt = True
        else:
            raise ValueError(f"split {split} not supported")
        self.df_anno = pd.read_csv(EMBED_ANNO_CSV_REDUCED)
        self.df_anno_full = pd.read_csv(EMBED_ANNO_CSV)
        df_legends = pd.read_csv(EMBED_LEGENDS_CSV)

        self.massshape_dict = {
            row["Code"]: row["Meaning"]
            for _, row in df_legends[
                df_legends["Header in export"] == "massshape"
            ].iterrows()
        }
        self.massdensity_dict = {
            row["Code"]: row["Meaning"]
            for _, row in df_legends[
                df_legends["Header in export"] == "massdens"
            ].iterrows()
        }
        self.calcfind_dict = {
            row["Code"]: row["Meaning"]
            for _, row in df_legends[
                df_legends["Header in export"] == "calcfind"
            ].iterrows()
        }
        self.calcdistri_dict = {
            row["Code"]: row["Meaning"]
            for _, row in df_legends[
                df_legends["Header in export"] == "calcdistri"
            ].iterrows()
        }

        # Only use 2D mammograms for now
        self.df = self.df[self.df[EMBED_IMAGE_TYPE_COL].isin(["2D"])]
        self.df[EMBED_PATH_COL] = self.df[EMBED_PATH_COL].apply(EMBED_PATH_TRANS_FUNC)
        
        # Only use screening images if screen_only is True
        if screen_only:
            screen_idx = self.df[EMBED_PROCEDURE_COL].apply(lambda x: x.lower().find('screen') > 0)
            self.df = self.df[screen_idx]
            # Clean up the magnification view and none CC/MLO view
            self.df = self.df[self.df['spot_mag'] != 0]
            self.df = self.df[self.df[EMBED_VIEW_COL].isin(['CC', 'MLO'])]

        if self.structural_cap or self.natural_cap:
            self.text_max_length = 144

        sub_train_set = split == "train" or train_sub_set
        if data_pct != 1.0 and sub_train_set:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        if split == "train":
            density_file = EMBED_TRAIN_PATH2DENSITY
        elif split == "valid":
            density_file = EMBED_VALID_PATH2DENSITY
        elif split == "test":
            density_file = EMBED_TEST_PATH2DENSITY
        else:
            raise ValueError(f"split {split} not supported")
        assert os.path.exists(density_file)
        self.path2density_pre = pickle.load(open(density_file, "rb"))

        if self.balanced_test:
            if self.pred_density:
                if ten_pct:
                    assert os.path.exists(EMBED_10PCT_DEN_TEST_PATH)
                    print("### Using balanced test set with 10% test examples...")
                    # Note this also contains the density label
                    self.balanced_test_path = pickle.load(
                        open(EMBED_10PCT_DEN_TEST_PATH, "rb")
                    )
                elif large_density:
                    assert os.path.exists(EMBED_BALANCED_LARGE_DEN_TEST_PATH)
                    print("### Using balanced test set with 4x2500 examples...")
                    # Note this also contains the density label
                    self.balanced_test_path = pickle.load(
                        open(EMBED_BALANCED_LARGE_DEN_TEST_PATH, "rb")
                    )
                else:
                    assert os.path.exists(EMBED_BALANCED_DEN_TEST_PATH)
                    print("### Using balanced test set with 4x500 examples...")
                    # Note this also contains the density label
                    self.balanced_test_path = pickle.load(
                        open(EMBED_BALANCED_DEN_TEST_PATH, "rb")
                    )
            else:
                if ten_pct:
                    assert os.path.exists(EMBED_10PCT_TEST_PATH)
                    print("### Using balanced test set with 10% test examples...")
                    self.balanced_test_path = pickle.load(
                        open(EMBED_10PCT_TEST_PATH, "rb")
                    )
                else:
                    assert os.path.exists(EMBED_BALANCED_TEST_PATH)
                    print("### Using balanced test set with 7x200 examples...")
                    self.balanced_test_path = pickle.load(
                        open(EMBED_BALANCED_TEST_PATH, "rb")
                    )
        else:
            self.balanced_test_path = None

        self.tokenizer = tokenizer

        self.filenames, self.path2sent, self.path2birads, self.path2density = self.load_text_data(split)

    def load_text_data(self, split):
        base_filename = f"{split}_mammo_clip_captions.pickle"
        if self.structural_cap:
            base_filename = base_filename.replace(".pickle", "_structural.pickle")
        elif self.simple_cap:
            base_filename = base_filename.replace(".pickle", "_simple.pickle")
        elif self.natural_cap:
            base_filename = base_filename.replace(".pickle", "_natural.pickle")
        if self.screen_only:
            base_filename = base_filename.replace(".pickle", "_screen.pickle")
        filepath = os.path.join(EMBED_DATA_DIR, base_filename)

        if not os.path.isfile(filepath):
            print(f"### Caption file {filepath} does not exist. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            print(f"### Loading captions from {filepath}...")
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # Some of the paths in the dataframe are not in the captions
        filenames = []
        path2birads = {}
        path2density = {}

        print("### extract label from captions...")
        for p, sentences in tqdm(path2sent.items()):
            # Only use the test image from balanced test set during test time
            if (
                self.split == "test"
                and self.balanced_test
                and p not in self.balanced_test_path.keys()
            ):
                continue
            if self.structural_cap:
                sent = sentences[-2]
            elif self.natural_cap or self.simple_cap or self.structural_cap:
                sent = sentences[-1]
            else:
                sent = sentences[-1].lower().replace('-', '')
            sent = sent.replace('bi rads', 'birads')
            assert 'birads' in sent
            if self.structural_cap or self.natural_cap:
                birads = re.findall(r"\bbirads\s\bcategory\s(\d+)", sent)[0]
            elif self.tabular_caption:
                sent = sent.replace(':', '')
                birads = re.findall(r"\bbirads\s(\d+)", sent)[0]
            else:
                birads = re.findall(r"\bbirads\s\bscore\s(\d+)", sent)[0]
            # skip birads 3 - 6 considering only screening image with 0, 1, 2
            if self.screen_only and int(birads) > 2:
                continue
            
            # if p not in self.path2density_pre.keys():
            #     print(f"### {p} not in density map")
            #     continue
            # Ignore male images
            density = self.path2density_pre[p] - 1
            if density == 4:
                density = 3
            #     continue
            path2density[p] = density

            path2birads[p] = int(birads)
            filenames.append(p)
        print(np.unique(list(path2birads.values()), return_counts=True))
        print(np.unique(list(path2density.values()), return_counts=True))
        return filenames, path2sent, path2birads, path2density

    def _create_captions_(self, row, meta_only=False):
        target_side = row[EMBED_SIDE_COL]
        anno_row = self.df_anno[self.df_anno[EMBED_SID_COL] == row[EMBED_SID_COL]]
        anno_full_row = self.df_anno_full[
            self.df_anno_full[EMBED_SID_COL] == row[EMBED_SID_COL]
        ]
        # Pick the correct side
        if target_side in anno_row[EMBED_FINDING_SIDE_COL].tolist():
            anno_row = anno_row[anno_row[EMBED_FINDING_SIDE_COL] == target_side]
            anno_full_row = anno_full_row[
                anno_full_row[EMBED_FINDING_SIDE_COL] == target_side
            ]
        elif "B" in anno_row[EMBED_FINDING_SIDE_COL].tolist():
            # Pick biliteral result otherwise
            anno_row = anno_row[anno_row[EMBED_FINDING_SIDE_COL] == "B"]
            anno_full_row = anno_full_row[anno_full_row[EMBED_FINDING_SIDE_COL] == "B"]
        try:
            # pick the case with highest BI-RADS
            all_asses = anno_row[EMBED_BIRADS_COL].to_list()
            all_birads = [
                EMBED_LETTER_TO_BIRADS[a]
                for a in all_asses
                if check_element_type(a, EMBED_LETTER_TO_BIRADS.keys())
            ]
            # If all screening image, prefer 0 case
            if np.max(all_birads) <= 2 and np.min(all_birads) == 0:
                idx = np.argmin(all_birads)
            else:
                idx = np.argmax(all_birads)
            anno_row = anno_row.iloc[idx]
            anno_full_row = anno_full_row.iloc[idx]
        except:
            anno_row = anno_row.iloc[0]
            anno_full_row = anno_full_row.iloc[0]
        # use the first annotation

        label_cnt = 0
        # if critical information is missing
        missing_info = False

        if self.structural_cap:
            captions = ""
            procedure = row[EMBED_PROCEDURE_COL]
            if check_element_type(procedure):
                captions += EMBED_PROCEDURE + procedure
                captions += "; "
                label_cnt += 1
            else:
                missing_info = True

            reason = EMBED_PROCEDURE2REASON_FUNC(procedure)
            if check_element_type(reason):
                captions += EMBED_REASON + reason
                captions += "; "
                label_cnt += 1
            else:
                missing_info = True

            age = anno_row[EMBED_AGE_COL]
            race = anno_row[EMBED_RACE_COL]
            ethnic = anno_row[EMBED_ETHNIC_COL]
            ethnic = (
                "Non-Hispanic or Latino" if ethnic != "Hispanic or Latino" else ethnic
            )
            # Check element type
            age = str(int(age)) if check_element_type(age) else "unknown"
            race = race if check_element_type(race) else "unknown"
            ethnic = ethnic if check_element_type(ethnic) else "unknown"
            # Replace the caption with information
            patient_cap = EMBED_PATIENT_INFO_CAPTION
            patient_cap = patient_cap.replace("{{RACE}}", race)
            patient_cap = patient_cap.replace("{{ETHNIC}}", ethnic)
            patient_cap = patient_cap.replace("{{AGE}}", age)
            captions += EMBED_PATIENT + patient_cap + " "
            label_cnt += 1

            image_type = row[EMBED_IMAGE_TYPE_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            # Check element type
            image_type = image_type if check_element_type(image_type) else "unknown"
            side = (
                EMBED_SIDES_DESC[side]
                if check_element_type(side, EMBED_SIDES_DESC.keys())
                else "unknown"
            )
            view = view if check_element_type(view) else "unknown"
            # Replace the caption with information
            image_cap = EMBED_IMAGE_INFO_CAPTION
            image_cap = image_cap.replace("{{IMAGE_TYPE}}", image_type)
            image_cap = image_cap.replace("{{SIDE}}", side)
            image_cap = image_cap.replace("{{VIEW}}", view)
            captions += EMBED_IMAGE + image_cap + " "
            label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info

            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += EMBED_DENSITY + EMBED_BREAST_COMPOSITION_CAPTION.replace(
                    "{{DENSITY}}", density_desc
                )
                if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                    captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            calc_find = False
            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                mass_info = EMBED_MASS_CAPTION[asses]
                shape_code = anno_full_row[EMBED_MASS_SHAPE_COL]
                density_code = anno_full_row[EMBED_MASS_DENSITY_COL]
                if check_element_type(
                    shape_code, self.massshape_dict.keys()
                ) and check_element_type(density_code, self.massdensity_dict.keys()):
                    mass_info += EMBED_MASS_EXTRA_CAPTION.replace(
                        "{{SHAPE}}", self.massshape_dict[shape_code]
                    ).replace("{{DENSITY}}", self.massdensity_dict[density_code])
                captions += EMBED_FINDINGS + EMBED_FINDS_CAPTION + mass_info + " "

                calc_find_code = anno_full_row[EMBED_CALC_FIND_COL]
                calc_distri_code = anno_full_row[EMBED_CALC_DIST_COL]
                if check_element_type(
                    calc_find_code, self.calcfind_dict.keys()
                ) and check_element_type(calc_distri_code, self.calcdistri_dict.keys()):
                    calc_info = EMBED_CALC_FINDS_CAPTION.replace(
                        "{{SHAPE}}", self.calcfind_dict[calc_find_code]
                    ).replace("{{DISTRI}}", self.calcdistri_dict[calc_distri_code])
                    captions += calc_info + " "
                    calc_find = True

                birads = EMBED_LETTER_TO_BIRADS[asses]
                impression_desc = EMBED_BIRADS_DESC[asses]
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", str(birads)
                ).replace("{{BIRADS_DESC}}", impression_desc)

                captions += EMBED_ASSESSMENT + EMBED_ASSESSMENT_CAPTION[asses]
                label_cnt += 1

                assert "{{" not in captions
                # dev
                # if calc_find:
                #     print(captions)
            else:
                missing_info = True
        elif self.natural_cap:
            captions = EMBED_NATURE_BASE_CAPTION

            procedure = row[EMBED_PROCEDURE_COL]
            reason = EMBED_PROCEDURE2REASON_FUNC(procedure)
            if check_element_type(reason):
                captions = captions.replace("{{REASON}}", reason)
            else:
                captions = captions.replace("{{REASON}}", "")

            age = anno_row[EMBED_AGE_COL]
            race = anno_row[EMBED_RACE_COL]
            ethnic = anno_row[EMBED_ETHNIC_COL]
            ethnic = (
                "Non-Hispanic or Latino" if ethnic != "Hispanic or Latino" else ethnic
            )
            # Check element type
            age = str(int(age)) if check_element_type(age) else "unknown"
            race = race if check_element_type(race) else "unknown"
            ethnic = ethnic if check_element_type(ethnic) else "unknown"
            patient_cap = EMBED_PATIENT_INFO_CAPTION
            patient_cap = patient_cap.replace("{{RACE}}", race)
            patient_cap = patient_cap.replace("{{ETHNIC}}", ethnic)
            patient_cap = patient_cap.replace("{{AGE}}", age)
            captions += patient_cap + " "
            label_cnt += 1

            image_type = row[EMBED_IMAGE_TYPE_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            # Check element type
            image_type = image_type if check_element_type(image_type) else "unknown"
            side = (
                EMBED_SIDES_DESC[side]
                if check_element_type(side, EMBED_SIDES_DESC.keys())
                else "unknown"
            )
            view = view if check_element_type(view) else "unknown"
            image_cap = EMBED_NATURE_IMAGE_CAPTION
            image_cap = image_cap.replace("{{SIDE}}", side)
            image_cap = image_cap.replace("{{VIEW}}", view)
            captions += image_cap + " "
            label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info

            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += EMBED_BREAST_COMPOSITION_CAPTION.replace(
                    "{{DENSITY}}", density_desc
                )
                if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                    captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                mass_info = EMBED_MASS_CAPTION[asses]
                shape_code = anno_full_row[EMBED_MASS_SHAPE_COL]
                density_code = anno_full_row[EMBED_MASS_DENSITY_COL]
                if check_element_type(
                    shape_code, self.massshape_dict.keys()
                ) and check_element_type(density_code, self.massdensity_dict.keys()):
                    mass_info += EMBED_MASS_EXTRA_CAPTION.replace(
                        "{{SHAPE}}", self.massshape_dict[shape_code]
                    ).replace("{{DENSITY}}", self.massdensity_dict[density_code])
                captions += EMBED_FINDS_CAPTION + mass_info + " "

                calc_find_code = anno_full_row[EMBED_CALC_FIND_COL]
                calc_distri_code = anno_full_row[EMBED_CALC_DIST_COL]
                if check_element_type(
                    calc_find_code, self.calcfind_dict.keys()
                ) and check_element_type(calc_distri_code, self.calcdistri_dict.keys()):
                    calc_info = EMBED_CALC_FINDS_CAPTION.replace(
                        "{{SHAPE}}", self.calcfind_dict[calc_find_code]
                    ).replace("{{DISTRI}}", self.calcdistri_dict[calc_distri_code])
                    captions += calc_info + " "

                birads = EMBED_LETTER_TO_BIRADS[asses]
                impression_desc = EMBED_BIRADS_DESC[asses]
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace(
                    "{{BIRADS}}", str(birads)
                ).replace("{{BIRADS_DESC}}", impression_desc)

                assert "{{" not in captions
            else:
                missing_info = True
        else:
            # Start with base caption
            captions = BREAST_BASE_CAPTION

            if not self.simple_cap:
                # provide extra side, view, density information
                side = row[EMBED_SIDE_COL]
                if check_element_type(side, EMBED_SIDES_DESC.keys()):
                    captions += BREAST_SIDE_CAPTION + EMBED_SIDES_DESC[side]
                    captions += " "
                    label_cnt += 1

                view = row[EMBED_VIEW_COL]
                if check_element_type(view):
                    captions += BREAST_VIEW_CAPTION + view
                    captions += " "
                    label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info

            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += (
                    BREAST_DENSITY_CAPTION
                    + str(int(density))
                    + ":"
                    + density_desc
                    + "."
                )
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                asses_desc = EMBED_BIRADS_DESC[asses]
                birads = EMBED_LETTER_TO_BIRADS[asses]
                captions += BREAST_BIRADS_CAPTION + str(birads) + ":" + asses_desc + "."
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

        return captions, label_cnt, missing_info

    def create_path_2_sent_mapping(self):
        sent_lens = []
        path2sent = {}
        for i, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # Find annotations for this image
            # Can be more than 1 annotations
            captions, label_cnt, missing_info = self._create_captions_(row)

            # Skip the image if there is no label
            if label_cnt == 0 or missing_info:
                continue

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                if len(tokens) <= 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 1:
                sent_lens.append(cnt)
                path2sent[row[EMBED_PATH_COL]] = study_sent

        sent_lens = np.array(sent_lens)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}] {len(sent_lens)}"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def random_mask(self, tokens, mask_ratio=0.1):
        # Unused
        return tokens
        masked_tokens = deepcopy(tokens)
        length = max(1, masked_tokens.shape[1] - 5)
        for i in range(1, length):
            if masked_tokens[0][i] == self.tokenizer.eos_token_id:
                break

            prob = random.random()
            if prob < mask_ratio:
                masked_tokens[0][i] = self.tokenizer.mask_token_id
        return tokens

    def get_caption(self, path, series_sents=None):
        if series_sents is None:
            series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.text_max_length,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        masked_ids = self.random_mask(tokens["input_ids"])
        tokens["masked_ids"] = masked_ids

        return tokens, x_len

    def get_birads_one_hot_label(self, index, get_full=False):
        num_classes = 3 if self.screen_only else len(EMBED_LETTER_TO_BIRADS)
        multi_hot_label = torch.zeros(num_classes)
        key = self.filenames[index]
        asses = self.path2birads[key]
        multi_hot_label[asses] = 1
        return multi_hot_label

    def get_density_one_hot_label(self, index, get_full=False):
        multi_hot_label = torch.zeros(len(EMBED_DENSITY_DESC) - 1)
        key = self.filenames[index]
        density = self.path2density[key]
        multi_hot_label[density] = 1
        return multi_hot_label


    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        key = GET_JPEG_PATH_FUNC(key)
        if self.aligned_mlo:
            aligned_key = GET_ALIGNED_MLO_FUNC(key)
            if os.path.exists(aligned_key):
                key = aligned_key
        imgs = get_imgs(key, self.imsize, self.transform)
        imgs = np.array(imgs).astype(np.float32)
        imgs -= imgs.min()
        imgs /= imgs.max()
        imgs = torch.tensor((imgs - self.mean) / self.std, dtype=torch.float32)
        imgs = imgs.unsqueeze(0)
        return {
            "images": imgs,
            "density": self.get_density_one_hot_label(index),
            "birads": self.get_birads_one_hot_label(index),
            "paths": key,
        }
