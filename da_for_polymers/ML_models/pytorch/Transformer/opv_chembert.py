import os
import pkg_resources
from argparse import ArgumentParser
from transformers import AutoModelForMaskedLM
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from torch.optim import SGD, Adam
from pytorch_lightning.callbacks import ModelCheckpoint

from da_for_polymers.ML_models.pytorch.data.OPV_Min.data import OPVDataModule

import ipdb

DATA_DIR = pkg_resources.resource_filename(
    "da_for_polymers", "data/process/master_da_for_polymers_from_min.csv"
)

TRAIN_FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/postprocess/train_frag_master.csv"
)

TRAIN_AUG_SMI_MASTER_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/augmentation/train_aug_master15.csv"
)

CHECKPOINT_DIR = pkg_resources.resource_filename(
    "da_for_polymers", "model_checkpoints/Transformer"
)

CHEMBERT_TOKENIZER = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/pytorch/Transformer/tokenizer_chembert/"
)

CHEMBERT = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/pytorch/Transformer/chembert/"
)

os.environ["WANDB_API_KEY"] = "95f67c3932649ca21ac76df3f88139dafacd965d"
os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# empty gpu cache
torch.cuda.empty_cache()

# initialize weights for model
def initialize_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data, gain=1)
        nn.init.constant_(model.bias.data, 0)


class TransformerModel(pl.LightningModule):
    """
    Class that will contain functions to access, modify, and use the ChemBERT transformers from HuggingFace
    """

    def __init__(self, pt_model, n_hidden, output_size, drop_prob, learning_rate):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            pt_model, output_hidden_states=True
        )
        embedding_dim = self.model.config.to_dict()["hidden_size"]
        self.loss = nn.MSELoss()
        self.linear = nn.Linear(embedding_dim, output_size)
        self.dropout = nn.Dropout(drop_prob)
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8
        )

    def forward(self, x):
        # [batch_size x seq_length x embedding_dim]
        # print("X: ", x.size())  # x = [44 x 260]
        ipdb.set_trace()
        embedded = self.model(x)  # [44 x 260 x 768]
        # print(embedded.size())
        # get last hidden layer
        embedded = embedded[len(embedded) - 1]
        # get first token of sequence
        embedded = embedded[:, 0, :]  # [44 x 1 x 768]
        # print(embedded)
        output = self.dropout(embedded)
        output = self.linear(output)[:, 0]  # [44 x 1]
        # print("final: ", output.size())
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat.size(), y.size())
        loss = self.loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat.size(), y.size())
        loss = self.loss(y_hat, y)
        self.log("val_loss_mse", loss, on_epoch=True)
        # corr_coef = np.corrcoef(y_hat.cpu(), y.cpu())[0, 1]
        # coef_determination = corr_coef ** 2
        # self.log("val_loss_r2", coef_determination, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)


def cli_main():
    pl.seed_everything(1)

    # ------------
    # wandb + sweep
    # ------------
    # wandb_logger = WandbLogger(project="OPV_ChemBERT", log_model=False, offline=False)
    # online
    # wandb_logger = WandbLogger(project="OPV_ChemBERT")

    # ------------
    # checkpoint + hyperparameter manual tuning
    # ------------
    n_hidden = 128
    n_embedding = 64
    drop_prob = 0.3
    learning_rate = 1e-2
    train_batch_size = 16
    val_batch_size = 16
    test_batch_size = 16

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--n_hidden", type=int, default=n_hidden)
    parser.add_argument("--n_output", type=int, default=1)
    parser.add_argument("--drop_prob", type=float, default=drop_prob)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    parser.add_argument("--n_embedding", type=int, default=n_embedding)
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR
    )  # DATA_DIR if not augmented smiles
    parser.add_argument("--train_batch_size", type=int, default=train_batch_size)
    parser.add_argument("--val_batch_size", type=int, default=val_batch_size)
    parser.add_argument("--test_batch_size", type=int, default=test_batch_size)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--accelerator", type=str, default="dp")
    parser.add_argument("--dataloader_num_workers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    # parser.add_argument("--logger", type=str, default=wandb_logger)
    args = parser.parse_args()

    # parse arguments using the terminal shell (for ComputeCanada purposes)
    suffix = (
        "/not_aug_smi_ChemBERT-{epoch:02d}-{val_loss_mse:.3f}"
        + "-drop_prob={}-lr={}-train_batch_size={}".format(
            args.drop_prob, args.learning_rate, args.train_batch_size,
        )
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_mse",
        filename=CHECKPOINT_DIR + suffix,
        save_top_k=2,
        mode="min",  # min if mse, max if r2
    )
    parser.add_argument("--callbacks", type=str, default=[checkpoint_callback])
    args = parser.parse_args()

    # pass args to wandb
    # wandb.init(project="OPV_ChemBERT", config=args)
    # config = wandb.config

    # ------------
    # data
    # ------------
    # smiles = True
    smiles = False
    aug_smiles = True
    # aug_smiles = False  # change to aug_smi (in suffix)
    aug_max = False  # change to aug_frag (in suffix)
    aug_pairs = False  # change to aug_pair_frag (in suffix)
    brics = False
    data_aug = False  # change to aug (in suffix)

    # for transformer
    chembert_model = CHEMBERT
    chembert_tokenizer = CHEMBERT_TOKENIZER

    data_module = OPVDataModule(
        data_dir=args.data_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.dataloader_num_workers,
        input=0,  # 0 - SMILES, 2 - SELFIES
        smiles=smiles,
        data_aug=data_aug,  # change to aug (in suffix)
        aug_smiles=aug_smiles,  # change to aug_smi (in suffix)
        aug_max=aug_max,  # change to aug_frag (in suffix)
        aug_pairs=aug_pairs,  # change to aug_pair_frag (in suffix)
        brics=brics,
        pt_model=chembert_model,
        pt_tokenizer=chembert_tokenizer,
    )
    # n_hidden states
    data_module.setup()
    data_module.prepare_data()

    # ------------
    # model
    # ------------

    transformer = TransformerModel(
        chembert_model,
        args.n_hidden,
        args.n_output,
        args.drop_prob,
        args.learning_rate,
    )
    transformer.apply(initialize_weights)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(transformer, data_module)


if __name__ == "__main__":
    cli_main()

# chembert_model = "seyonec/ChemBERTa_zinc250k_v2_40k"
# tokenizer = AutoTokenizer.from_pretrained(CHEMBERT_TOKENIZER)
# tokenizer.padding_side = "right"
# tokenizer.save_pretrained("./tokenizer_chembert")
# print(tokenizer) # get dictionary
# pt_model = AutoModelForMaskedLM.from_pretrained(CHEMBERT, output_hidden_states=True)
# pt_model.save_pretrained("./chembert")
# batch = ["CCCCCCCCCCCCCCCCCCCCCCCCC.CCCCCCCCCC.CCC", "CCCCC.CCCCC.CCCCC"]
# pt_batch = tokenizer(batch, padding=True, return_tensors="pt")
# print(pt_batch)
# # print(pt_batch["input_ids"].size())  # torch.Size([1, 8])
# pt_output = pt_model(**pt_batch)
# print("output: ", pt_output)
# print(pt_output[1])
# feature_vector = list(pt_output.hidden_states)[0]
# print(feature_vector)
