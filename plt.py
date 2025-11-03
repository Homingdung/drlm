import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

#plt.plot(df["time"], df["helicity_c"], label="crossHelicity")
plt.plot(df["time"], df["helicity_m"], label="magneticHelicity")
#plt.plot(df["time"], df["helicity_f"], label="fluidHelicity")
#plt.plot(df["time"], df["energy"], label="Energy")

plt.xlabel("Time")
plt.ylabel("Value")
#plt.title("XXX vs Time")
plt.legend()
#plt.grid(True)

plt.savefig("result.png", dpi=300)
plt.show()
