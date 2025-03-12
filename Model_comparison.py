# -*- coding: utf-8 -*-
"""
分割表をランダムにサンプリングし，各モデルの挙動を比較するためのコード
"""

import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import itertools

"""## サンプリング，モデルの関数"""

def CS(conts, threshold, is_gene, loops):
    rng = np.random.default_rng()  # 乱数ジェネレーター
    power = np.zeros((loops, 2))

    # ループを実行して条件を満たす乱数を生成
    for i in range(loops):
      power0 = rng.uniform(0 + 1e-10, 1)
      power1 = rng.uniform(0 + 1e-10, threshold)
      # power1 = threshold
      power[i] = [power0,power1]

    a, b, c, d = conts

    # power[:, 0] 原因と結果のw
    # power[:, 1] 背景と結果のw

    if is_gene:
      probs1 = [
          (1 - (1 - power[:, 0]) * (1 - power[:, 1])),# P(E=1|C=1)
          (1 - power[:, 0]) * (1 - power[:, 1]),# P(E=0|C=1)
          power[:, 1],# P(E=1|C=0)
          (1 - power[:, 1]),# P(E=0|C=0)
      ]
    else:
      probs1 = [
          power[:, 1] - (power[:, 0] * power[:, 1]),# P(E=1|C=1)
          1 - (power[:, 1] - (power[:, 0] * power[:, 1])),# P(E=0|C=1)
          power[:, 1],# P(E=1|C=0)
          1 - power[:, 1],# P(E=0|C=0)
      ]

    probs0 = [
        power[:, 1],# P(E=1|C=1)
        (1 - power[:, 1]),# P(E=0|C=1)
        power[:, 1],# P(E=1|C=0)
        (1 - power[:, 1]),# P(E=0|C=0)
    ]

    loglike1 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs1).T, axis=1)
    like1 = sum(np.exp(loglike1)) * (1/loops)

    loglike0 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs0).T, axis=1)
    like0 = sum(np.exp(loglike0)) * (1/loops)

    logscore = np.log(like1/like0)
    return logscore

def UCS(conts, threshold, is_gene, loops):
    rng = np.random.default_rng()  # 乱数ジェネレーター
    power = np.zeros((loops, 3))

    # ループを実行して条件を満たす乱数を生成
    for i in range(loops):
      power0 = rng.uniform(0 + 1e-10, 1)
      power1 = rng.uniform(0 + 1e-10, threshold)
      power2 = rng.uniform(0 + 1e-10, threshold)
      power[i] = [power0,power1,power2]
    a, b, c, d = conts
    # power[:, 0] 原因と結果のw
    # power[:, 1] 背景と原因のw
    # power[:, 2] 背景と結果のw
    if is_gene:
      # z = /(np.sqrt(power[:, 1] * power[:, 2] * (1 - (1 - power[:, 0]) * (1 - power[:, 1])) * (1 - (1 - power[:, 0]) * (1 - power[:, 2]))) + np.sqrt(power[:, 1] * (1 - power[:, 2]) * (power[:, 1] * ((1 - power[:, 0]) * (1 - power[:, 2])))) + np.sqrt((1 - power[:, 1]) * power[:, 2] * (power[:, 2] * ((1 - power[:, 0]) * (1 - power[:, 1])))) + (1 - power[:, 1]) * (1 - power[:, 2]))
      probs1 = [
          np.sqrt(power[:, 1] * power[:, 2] * (1 - (1 - power[:, 0]) * (1 - power[:, 1])) * (1 - (1 - power[:, 0]) * (1 - power[:, 2])))/(np.sqrt(power[:, 1] * power[:, 2] * (1 - (1 - power[:, 0]) * (1 - power[:, 1])) * (1 - (1 - power[:, 0]) * (1 - power[:, 2]))) + np.sqrt(power[:, 1] * (1 - power[:, 2]) * (power[:, 1] * ((1 - power[:, 0]) * (1 - power[:, 2])))) + np.sqrt((1 - power[:, 1]) * power[:, 2] * (power[:, 2] * ((1 - power[:, 0]) * (1 - power[:, 1])))) + (1 - power[:, 1]) * (1 - power[:, 2])),
          np.sqrt(power[:, 1] * (1 - power[:, 2]) * (power[:, 1] * ((1 - power[:, 0]) * (1 - power[:, 2]))))/(np.sqrt(power[:, 1] * power[:, 2] * (1 - (1 - power[:, 0]) * (1 - power[:, 1])) * (1 - (1 - power[:, 0]) * (1 - power[:, 2]))) + np.sqrt(power[:, 1] * (1 - power[:, 2]) * (power[:, 1] * ((1 - power[:, 0]) * (1 - power[:, 2])))) + np.sqrt((1 - power[:, 1]) * power[:, 2] * (power[:, 2] * ((1 - power[:, 0]) * (1 - power[:, 1])))) + (1 - power[:, 1]) * (1 - power[:, 2])),
          np.sqrt((1 - power[:, 1]) * power[:, 2] * (power[:, 2] * ((1 - power[:, 0]) * (1 - power[:, 1]))))/(np.sqrt(power[:, 1] * power[:, 2] * (1 - (1 - power[:, 0]) * (1 - power[:, 1])) * (1 - (1 - power[:, 0]) * (1 - power[:, 2]))) + np.sqrt(power[:, 1] * (1 - power[:, 2]) * (power[:, 1] * ((1 - power[:, 0]) * (1 - power[:, 2])))) + np.sqrt((1 - power[:, 1]) * power[:, 2] * (power[:, 2] * ((1 - power[:, 0]) * (1 - power[:, 1])))) + (1 - power[:, 1]) * (1 - power[:, 2])),
          (1 - power[:, 1]) * (1 - power[:, 2])/(np.sqrt(power[:, 1] * power[:, 2] * (1 - (1 - power[:, 0]) * (1 - power[:, 1])) * (1 - (1 - power[:, 0]) * (1 - power[:, 2]))) + np.sqrt(power[:, 1] * (1 - power[:, 2]) * (power[:, 1] * ((1 - power[:, 0]) * (1 - power[:, 2])))) + np.sqrt((1 - power[:, 1]) * power[:, 2] * (power[:, 2] * ((1 - power[:, 0]) * (1 - power[:, 1])))) + (1 - power[:, 1]) * (1 - power[:, 2])),
      ]
    else:
      # z = / (np.sqrt((power[:, 1] * power[:, 2]) * ((power[:, 1] * (1 - power[:, 0])) * (power[:, 2] * (1 - power[:, 0])))) + np.sqrt((power[:, 1] * (1 - power[:, 2])) * (power[:, 1] * (1 - (power[:, 2] * (1 - power[:, 0]))))) + np.sqrt(((1 - power[:, 1]) * power[:, 2]) * ((1 - (power[:, 1] * (1 - power[:, 0]))) * power[:, 2])) + (1 - power[:, 1]) * (1 - power[:, 2]))
      probs1 = [
          np.sqrt((power[:, 1] * power[:, 2]) * ((power[:, 1] * (1 - power[:, 0])) * (power[:, 2] * (1 - power[:, 0]))))  / (np.sqrt((power[:, 1] * power[:, 2]) * ((power[:, 1] * (1 - power[:, 0])) * (power[:, 2] * (1 - power[:, 0])))) + np.sqrt((power[:, 1] * (1 - power[:, 2])) * (power[:, 1] * (1 - (power[:, 2] * (1 - power[:, 0]))))) + np.sqrt(((1 - power[:, 1]) * power[:, 2]) * ((1 - (power[:, 1] * (1 - power[:, 0]))) * power[:, 2])) + (1 - power[:, 1]) * (1 - power[:, 2])),
          np.sqrt((power[:, 1] * (1 - power[:, 2])) * (power[:, 1] * (1 - (power[:, 2] * (1 - power[:, 0])))))  / (np.sqrt((power[:, 1] * power[:, 2]) * ((power[:, 1] * (1 - power[:, 0])) * (power[:, 2] * (1 - power[:, 0])))) + np.sqrt((power[:, 1] * (1 - power[:, 2])) * (power[:, 1] * (1 - (power[:, 2] * (1 - power[:, 0]))))) + np.sqrt(((1 - power[:, 1]) * power[:, 2]) * ((1 - (power[:, 1] * (1 - power[:, 0]))) * power[:, 2])) + (1 - power[:, 1]) * (1 - power[:, 2])),
          np.sqrt(((1 - power[:, 1]) * power[:, 2]) * ((1 - (power[:, 1] * (1 - power[:, 0]))) * power[:, 2]))  / (np.sqrt((power[:, 1] * power[:, 2]) * ((power[:, 1] * (1 - power[:, 0])) * (power[:, 2] * (1 - power[:, 0])))) + np.sqrt((power[:, 1] * (1 - power[:, 2])) * (power[:, 1] * (1 - (power[:, 2] * (1 - power[:, 0]))))) + np.sqrt(((1 - power[:, 1]) * power[:, 2]) * ((1 - (power[:, 1] * (1 - power[:, 0]))) * power[:, 2])) + (1 - power[:, 1]) * (1 - power[:, 2])),
          (1 - power[:, 1]) * (1 - power[:, 2])  / (np.sqrt((power[:, 1] * power[:, 2]) * ((power[:, 1] * (1 - power[:, 0])) * (power[:, 2] * (1 - power[:, 0])))) + np.sqrt((power[:, 1] * (1 - power[:, 2])) * (power[:, 1] * (1 - (power[:, 2] * (1 - power[:, 0]))))) + np.sqrt(((1 - power[:, 1]) * power[:, 2]) * ((1 - (power[:, 1] * (1 - power[:, 0]))) * power[:, 2])) + (1 - power[:, 1]) * (1 - power[:, 2])),
      ]
    probs0 = [
        power[:, 1] * power[:, 2],
        power[:, 1] * (1 - power[:, 2]),
        (1 - power[:, 1]) * power[:, 2],
        (1 - power[:, 1]) * (1 - power[:, 2]),
    ]

    loglike1 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs1).T, axis=1)
    like1 = sum(np.exp(loglike1)) * (1/loops)

    loglike0 = np.sum((np.ones((loops, 1)) * np.array(conts)) * np.log(probs0).T, axis=1)
    like0 = sum(np.exp(loglike0)) * (1/loops)

    logscore = np.log(like1/like0)
    return logscore

def sample_from_distribution(a, b, c, d, n_samples):
    # 確率のリスト
    probabilities = [a, b, c, d]

    # 値のリスト
    values = ['A', 'B', 'C', 'D']

    # サンプリングを実行
    samples = np.random.choice(values, size=n_samples, p=probabilities)

    # 各値が得られた回数をカウント
    counts = {value: 0 for value in values}
    for sample in samples:
        counts[sample] += 1

    # カウントを配列に変換
    count_array = [counts[value] for value in values]

    return count_array

"""## サンプリング"""

sample_sizes = [7]
thresholds = [i / 10 for i in range(1, 11)]  # 0.0 to 1.0 in 0.1 increments
loop_count = 10000
df = pd.DataFrame()
random.seed(time.time())
for sample_size in sample_sizes:
  for i in range(10):
    a = random.randint(0, sample_size)
    b = random.randint(0, sample_size)
    c = random.randint(0, sample_size)
    d = random.randint(0, sample_size)

    results = {'sample_size': a + b + c + d, 'a': a, 'b': b, 'c': c, 'd': d}
    for threshold in thresholds:
        results[f'cs_{threshold}'] = CS([a, b, c, d], threshold, True, loop_count)
        results[f'ucs_{threshold}'] = UCS([a, b, c, d], threshold, True, loop_count)

    dice = (2*a) / (2*a + b + c) if (2*a + b + c) != 0 else None
    phi = (a * d - b * c) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d)) \
        if (a + b) * (a + c) * (b + d) * (c + d) != 0 else None
    deltap = (a / (a + b) - c / (c + d)) \
        if (a + b) != 0 and (c + d) != 0 else None
    pe_c = a / (a + b) if (a + b) != 0 else None

    results.update({'dice_value': dice, 'phi_value': phi, 'deltap_value': deltap, 'pe_c_value': pe_c})
    df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

df

df.corr()

# tmp_df = df['cs_value','cs_unif_value']
tmp_df = df.loc[:,['cs_0.1','cs_0.3','cs_1.0', 'ucs_0.1', 'ucs_0.3', 'ucs_1.0','dice_value','phi_value','deltap_value', 'pe_c_value']]
tmp_df.corr()

tmp_df.corr(method='spearman')

"""##プロット 二つのセル毎の変化"""

variables = ['a', 'b', 'c', 'd']

# すべての2変数の組み合わせを取得
combinations = list(itertools.combinations(variables, 2))

"""Dice 係数"""

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='dice_value', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean dice_value ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()

"""pARIs

UCS 0.1
"""

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='ucs_0.1', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean ucs_0.1 ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='ucs_0.3', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean ucs_0.3 ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='ucs_0.5', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean ucs_0.5 ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='ucs_1.0', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean ucs_1.0 ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()

"""CS"""

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='cs_0.1', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean cs_0.1 ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='cs_0.3', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean cs_0.3 ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='cs_0.5', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean cs_0.5 ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()

# ヒートマップを描画
plt.figure(figsize=(12, 10))
for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(3, 3, i)
    pivot_table = df.pivot_table(index=x, columns=y, values='cs_1.0', aggfunc='mean')
    sns.heatmap(pivot_table.sort_index(ascending=False), annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title(f'Mean cs_1.0 ({x} vs {y})')
    plt.xlabel(y)
    plt.ylabel(x)

plt.tight_layout()
plt.show()