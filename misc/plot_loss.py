import pandas as pd
import matplotlib.pyplot as plt


loss_path = './data/weights/loss_per_epoch.csv'
df = pd.read_csv(loss_path)
plt.figure(figsize=(10, 6))
plt.plot(df['Epoca'], df['Loss'], linestyle='-', color='blue')
plt.title('Loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
#    plt.xticks(epoch_list)
plt.tight_layout()
# Salvar o gráfico como um arquivo PNG
plt.savefig('./data/weights/loss_per_epoch.png')
