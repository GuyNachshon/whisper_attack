"""
Evaluation script supporting adversarial attacks.
Similar to the training script without the brain.fit() call, with one key difference:
To support transferred attacks or attacks conducted on multiple models like MGAA,
the model hparams files was decoupled from the main hparams file.

hparams contains target_brain_class and target_brain_hparams_file arguments,
which are used to load corresponding brains, modules and pretrained parameters.
Optional source_brain_class and source_brain_hparams_file can be specified to transfer
the adversarial perturbations. Each of them can be specified as a (nested) list, in which case
the brain will be an EnsembleASRBrain object.

Example:
python run_attack.py attack_configs/pgd/attack.yaml\
     --root=/path/to/data/and/results/folder\
     --auto_mix_prec`
"""
import os
import sys
from pathlib import Path
import logging
import shutil
import urllib.request
import tarfile
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from tqdm import tqdm
import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain
from robust_speech.adversarial.utils import TargetGeneratorFromFixedTargets

logging.basicConfig(level=logging.INFO)

def check_disk_space(path, required_gb):
    """Check if there's enough disk space available."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free // (2**30)  # Convert to GB
    return free_gb >= required_gb

def download_librispeech(download_path):
    """Download and extract LibriSpeech dataset chunks with proper folder structure."""
    dataset_path = Path(download_path)
    download_url = "http://www.openslr.org/resources/12/test-clean.tar.gz"
    
    # Create base directory
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Download the dataset
    tar_path = dataset_path / "test-clean.tar.gz"
    logging.info(f"Downloading LibriSpeech dataset to {dataset_path}")
    urllib.request.urlretrieve(download_url, tar_path)
    
    # Create a temporary directory for initial extraction
    temp_extract_dir = dataset_path / "temp_extract"
    temp_extract_dir.mkdir(exist_ok=True)
    
    # Extract the dataset to temp directory first
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=temp_extract_dir)
    
    # Move the contents to the correct location
    # LibriSpeech archive contains a 'LibriSpeech' directory inside
    extracted_librispeech = temp_extract_dir / "LibriSpeech" / "test-clean"
    target_dir = dataset_path / "test-clean"
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # Move the test-clean directory to the correct location
    shutil.move(str(extracted_librispeech), str(dataset_path))
    
    # Clean up
    shutil.rmtree(temp_extract_dir)
    os.remove(tar_path)
    
    # Verify the structure
    if not (dataset_path / "test-clean").exists():
        raise RuntimeError(f"Failed to create proper LibriSpeech structure in {dataset_path}")
        
    logging.info(f"LibriSpeech dataset downloaded and extracted to {dataset_path} with correct structure")

def read_brains(
    brain_classes,
    brain_hparams,
    attacker=None,
    run_opts={},
    overrides={},
    tokenizer=None,
):
    logging.info(f"Reading brains: {brain_classes} {brain_hparams}")
    if isinstance(brain_classes, list):
        brain_list = []
        assert len(brain_classes) == len(brain_hparams)
        for bc, bf in tqdm(zip(brain_classes, brain_hparams), total=len(brain_classes), desc="Reading brain models"):
            logging.info(f"Loading brain model: {bc}")
            br = read_brains(
                bc, bf, run_opts=run_opts, overrides=overrides, tokenizer=tokenizer
            )
            brain_list.append(br)
            logging.info(f"Successfully loaded brain model: {bc}")
        brain = rs.adversarial.brain.EnsembleASRBrain(brain_list)
        logging.info(f"Created ensemble brain with {len(brain_list)} models")
    else:
        if isinstance(brain_hparams, str):
            logging.info(f"Loading brain hyperparameters from: {brain_hparams}")
            with open(brain_hparams) as fin:
                brain_hparams = load_hyperpyyaml(fin, overrides)
        checkpointer = (
            brain_hparams["checkpointer"] if "checkpointer" in brain_hparams else None
        )
        logging.info("Initializing brain model...")
        brain = brain_classes(
            modules=brain_hparams["modules"],
            hparams=brain_hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
            attacker=attacker,
        )
        if "pretrainer" in brain_hparams:
            logging.info("Loading pretrained parameters for brain model...")
            run_on_main(brain_hparams["pretrainer"].collect_files)
            brain_hparams["pretrainer"].load_collected(
                device=run_opts["device"])
            logging.info("Successfully loaded pretrained parameters")
        brain.tokenizer = tokenizer
    return brain


def evaluate(hparams_file, run_opts, overrides):
    logging.info(f"Starting evaluation with hparams file: {hparams_file}")
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        logging.info(f"Loaded hyperparameters with overrides: {overrides}")

    # Ensure root path is set
    if hparams["root"] == "!PLACEHOLDER" or not hparams["root"]:
        hparams["root"] = os.getcwd()  # Use current working directory
        logging.info(f"Setting root directory to current working directory: {hparams['root']}")
        
    # Update all dependent paths relative to root
    hparams["data_folder"] = os.path.join(hparams["root"], "data/LibriSpeech")
    hparams["csv_folder"] = os.path.join(hparams["data_folder"], "csv")
    hparams["output_folder"] = os.path.join(hparams["root"], "attacks", hparams["attack_name"], 
                                           f"whisper-{hparams['model_label']}", str(hparams["seed"]))
    
    # Ensure folders exist
    os.makedirs(hparams["data_folder"], exist_ok=True)
    os.makedirs(hparams["csv_folder"], exist_ok=True)
    os.makedirs(hparams["output_folder"], exist_ok=True)
    
    # Set CSV paths - ensure it's a list
    csv_prefix = "test-clean"
    csv_path = os.path.join(hparams["csv_folder"], f"{csv_prefix}.csv")
    hparams["test_csv"] = [csv_path]  # Must be a list for the dataio_prepare function
    
    logging.info(f"Using data folder: {hparams['data_folder']}")
    logging.info(f"Using CSV folder: {hparams['csv_folder']}")
    logging.info(f"Using CSV files: {hparams['test_csv']}")
    logging.info(f"Using output folder: {hparams['output_folder']}")

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    logging.info(f"Created experiment directory: {hparams['output_folder']}")

    # Check and download dataset if needed
    if hparams.get("download_data", False) or not os.path.exists(hparams["data_folder"]):
        logging.info("Checking LibriSpeech dataset...")
        data_folder = Path(hparams["data_folder"])
        if not all((data_folder / split).exists() for split in hparams["test_splits"]):
            logging.info("Dataset not found. Starting download...")
            data_folder.mkdir(parents=True, exist_ok=True)
            download_librispeech(str(data_folder))
            logging.info("Dataset download completed")
        else:
            logging.info("Dataset already exists")

    if "pretrainer" in hparams:
        logging.info("Loading pretrained parameters...")
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])
        logging.info("Pretrained parameters loaded successfully")

    # Dataset prep (parsing Librispeech)
    logging.info("Preparing dataset...")
    prepare_dataset = hparams["dataset_prepare_fct"]

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_dataset,
        kwargs={
            "data_folder": hparams["data_folder"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["csv_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    dataio_prepare = hparams["dataio_prepare_fct"]
    logging.info("Preparing data IO and tokenization...")
    if "tokenizer_builder" in hparams:
        logging.info(f"Building tokenizer: {hparams['tokenizer_name']}")
        tokenizer = hparams["tokenizer_builder"](hparams["tokenizer_name"])
        hparams["tokenizer"] = tokenizer
    else:
        tokenizer=hparams["tokenizer"]

    # here we create the datasets objects as well as tokenization and encoding
    _, _, test_datasets, _, _, tokenizer = dataio_prepare(hparams)
    logging.info(f"Created {len(test_datasets)} test datasets")

    source_brain = None
    if "source_brain_class" in hparams:  # loading source model
        logging.info("Loading source model...")
        source_brain = read_brains(
            hparams["source_brain_class"],
            hparams["source_brain_hparams_file"],
            run_opts=run_opts,
            overrides={"root": hparams["root"]},
            tokenizer=tokenizer,
        )
    attacker = hparams["attack_class"]
    if source_brain and attacker:
        # instanciating with the source model if there is one.
        # Otherwise, AdvASRBrain will handle instanciating the attacker with
        # the target model.
        if isinstance(source_brain, rs.adversarial.brain.EnsembleASRBrain):
            if "source_ref_attack" in hparams:
                source_brain.ref_attack = hparams["source_ref_attack"]
            if "source_ref_train" in hparams:
                source_brain.ref_train = hparams["source_ref_train"]
            if "source_ref_valid_test" in hparams:
                source_brain.ref_valid_test = hparams["source_ref_valid_test"]
        attacker = attacker(source_brain)

    # Target model initialization
    target_brain_class = hparams["target_brain_class"]
    target_hparams = (
        hparams["target_brain_hparams_file"]
        if hparams["target_brain_hparams_file"]
        else hparams
    )
    if source_brain is not None and target_hparams == hparams["source_brain_hparams_file"]:
        # avoid loading the same network twice
        sc_brain = source_brain
        sc_class = target_brain_class
        if isinstance(source_brain, rs.adversarial.brain.EnsembleASRBrain):
            sc_brain = source_brain.asr_brains[source_brain.ref_valid_test]
            sc_class = target_brain_class[source_brain.ref_valid_test]
        target_brain = sc_class(
            modules=sc_brain.modules,
            hparams=sc_brain.hparams.__dict__,
            run_opts=run_opts,
            checkpointer=None,
            attacker=attacker,
        )
        target_brain.tokenizer = tokenizer
    else:
        target_brain = read_brains(
            target_brain_class,
            target_hparams,
            attacker=attacker,
            run_opts=run_opts,
            overrides={"root": hparams["root"]},
            tokenizer=tokenizer,
        )
    target_brain.logger = hparams["logger"]
    target_brain.hparams.train_logger = hparams["logger"]
    #attacker.other_asr_brain = target_brain
    target = None
    if "target_generator" in hparams:
        target = hparams["target_generator"]
    elif "target_sentence" in hparams:
        target = TargetGeneratorFromFixedTargets(
            target=hparams["target_sentence"])
    load_audio = hparams["load_audio"] if "load_audio" in hparams else None
    save_audio_path = hparams["save_audio_path"] if hparams["save_audio"] else None
    # Evaluation
    total_datasets = len(test_datasets.keys())
    for k in tqdm(test_datasets.keys(), desc="Evaluating datasets", total=total_datasets):
        logging.info(f"\nEvaluating dataset: {k}")
        target_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        logging.info(f"WER will be saved to: {target_brain.hparams.wer_file}")
        target_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            load_audio=load_audio,
            save_audio_path=save_audio_path,
            sample_rate=hparams["sample_rate"],
            target=target,
        )
        logging.info(f"Completed evaluation for dataset: {k}")


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(f"hparams_file: {hparams_file}")
    print(f"run_opts: {run_opts}")
    print(f"overrides: {overrides}")
    print("\n******\ndone")
    raise Exception("stop here")
    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    evaluate(hparams_file, run_opts, overrides)
