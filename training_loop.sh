echo "Epoch 1/60:"
SLURM=1 python SFC_backprop/SFC_backprop_main.py --weight_mode rand_He
for i in {2..60}
do
  echo "Epoch $i/60:"
  SLURM=1 python SFC_backprop/SFC_backprop_main.py --weight_mode restore
done
