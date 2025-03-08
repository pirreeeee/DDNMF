# DDNMF

此方法結合了非負矩陣分解 (NMF) 和遞迴神經網絡 (RNN)，提出深度動態神經網路 (Dynamic Deep NMF)，來捕捉動態模式。

## 目標

我們的目標是在非監督式學習(unsupervised learning)中，目前有許多人會花費大量成本人工標記無標籤資料，坊間也有許多AI標記公司提供標記服務，以利於將資料用於訓練模型，但對於中小型及新創用戶來說，這樣的成本並非人人可以承擔，因此我們的模型可以在沒有標籤的情況下，獲得準確度相對高的結果，以利於初步的資料分析。
