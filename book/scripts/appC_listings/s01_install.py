import rnel
from rnel import neutro_credal as nc

print("rnel version:", rnel.__version__)
print("rnel.neutro_credal exports", len(nc.__all__), "names:")
for i in range(0, len(nc.__all__), 5):
    print("  " + ", ".join(nc.__all__[i:i + 5]))
