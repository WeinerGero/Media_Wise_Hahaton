import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale, scale

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TimeSeriesDataSet, TemporalFusionTransformer, DeepAR, NHiTS
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE, MAE, SMAPE
from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss, NormalDistributionLoss, MQF2DistributionLoss

from torchmetrics.functional import r2_score, mean_absolute_percentage_error

def get_known_values(table, header_row_idx=4):
    """
    Убираем из таблицы заголовок, берём в качестве заголовка пятую строку. 
    Если есть повторения, то добавляем к ним индекс, чтобы различать.
    В проде это можно сделать при помощи языковой модели.
    """
    df_wo_header = table[header_row_idx:]
    namelist = [v for i, v in df_wo_header.iloc[0].reset_index(drop=True).items()]
    namelist_wo_dups = [v + str(namelist[:i].count(v) + 1) if namelist.count(v) > 1 else v for i, v in enumerate(namelist)]
    
    df2 = df_wo_header
    df2.columns = namelist_wo_dups
    df = df2.drop([header_row_idx]).reset_index(drop=True)
    df.insert(0, 'time_idx', df.index)
    df_known = df[:244].drop(columns=['год', 'неделя'])
    return df_known

def pre_work_w_NaN(table, nan_percent_to_drop = 0.8):
    table = table.replace({'0':np.nan, 0:np.nan})
    labels_to_drop = [x for x in table.columns if table[x].isna().sum()/len(table.index) > nan_percent_to_drop]
    labels_to_interpolate = [x for x in table.columns if 0 < table[x].isna().sum()/len(table.index) <= nan_percent_to_drop]

    df_filled = table.copy()
    df_filled = df_filled.drop(columns=labels_to_drop)
    df_filled[labels_to_interpolate] = df_filled[labels_to_interpolate].interpolate(method='linear').fillna(0)
    return df_filled

def scale_data(table):
    cols = table.select_dtypes(np.number).columns
    table[cols] = minmax_scale(table[cols])
    table.time_idx = table.index.astype(int)
    return table

def filter_by_correlations(table, lower_threshold = 0.2, upper_threshold = 0.7, target='Продажи, рубли1'):
    correlations = table.corr()[target]
    mask1 = (correlations[:] < -upper_threshold) | (correlations[:] > upper_threshold)
    mask2 = (correlations[:] > -lower_threshold) & (correlations[:] < lower_threshold)
    np.bitwise_or(mask1, mask2)
    
    correlated = correlations.drop(correlations[mask1 | mask2].index)
    return correlated

data = pd.read_excel('https://docs.google.com/spreadsheets/d/1kqfX08UG3Y5eIS7d8siQFGu8M1x8c5Vt/export')
df_no_nans = scale_data(pre_work_w_NaN(get_known_values(data)))
corrs = filter_by_correlations(df_no_nans)
ax = corrs.plot(kind='barh', figsize=(10,10))
plt.savefig('correlations.png')

from pytorch_forecasting import TimeSeriesDataSet
def build_datasets(table, pred_length = 28):
    encoder_length = 56
    training_cutoff = table['time_idx'].max() - pred_length
    train_dataset = TimeSeriesDataSet(
        table[lambda x: x.time_idx <= training_cutoff],
        time_idx = 'time_idx',
        target = 'Продажи, упаковки',
        group_ids = ['group'],
        min_encoder_length = encoder_length, # = то, на сколько значений НАЗАД мы смотрим
        max_encoder_length = encoder_length,
        min_prediction_length = pred_length, # = то, на сколько значений ВПЕРЕД мы прогнозируем
        max_prediction_length = pred_length,
        time_varying_unknown_reals = ['Продажи, упаковки'],
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(train_dataset, table, predict=True, stop_randomization=True, min_prediction_idx=training_cutoff + 1)
    return train_dataset, validation_dataset

def build_dataloaders(train_dataset, validation_dataset): 
    batch_size = 16  # set this between 32 to 128
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    train_evaluation_dataloader = train_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=2, batch_sampler="synchronized")
    validation_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=2)
    return train_dataloader, validation_dataloader

def train_NHITS(train_dataset, train_dataloader, validation_dataloader, pred_length = 28):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    lr_logger = LearningRateMonitor(logging_interval='step', log_momentum=True)  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='auto',
        enable_model_summary=True,
        #gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        enable_progress_bar=True, #val_check_interval=10,
        #log_every_n_steps=10,
        logger=logger
    )
    net = NHiTS.from_dataset(
         train_dataset,
         learning_rate=0.05,
         hidden_size=128,
         loss=MQF2DistributionLoss(prediction_length=pred_length),
         optimizer="AdamW"
    )
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)
    best_model = NHiTS.load_from_checkpoint(best_model_path)

    return best_model

def evaluate(predictions, answers):
    rmse = RMSE()(predictions, answers).item()
    smape = SMAPE()(predictions, answers).item()
    mae = MAE()(predictions, answers).item()

    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.3f}")
    print(f"SMAPE: {smape:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"===============")

cols = ['time_idx', 'Начало нед', 'Продажи, упаковки'] + corrs.index.to_list()
df_corr = df_no_nans[cols]

# это уже для датасета делаем
df_corr['group'] = 0
df_corr['time_idx'] = df_no_nans.index

train_ds, val_ds = build_datasets(df_corr)
train_dl, val_dl = build_dataloaders(train_ds, val_ds)
model = train_NHITS(train_ds, train_dl, val_dl)

predictions = model.predict(validation_dataloader)
# Запишем ответы из валидации в один тензор, чтобы быстрее считать метрики
val_answers = torch.cat([y[0] for x, y in iter(validation_dataloader)])

print("Our model predictions:")
evaluate(predictions, val_answers)



