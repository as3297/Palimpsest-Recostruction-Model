#!/bin/bash
DATANAME=archi
LOGDIR='C:/Data/PhD/bss_gan_palimpsest/training/InvNet'
CONV="1d"
RESTOREDPATH_UTB='C:/Data/PhD/bss_gan_palimpsest/training/DCGAN/greek_mode_utb_background_antishort_30lutb_16lb_32gutb_8gut_16dut_8dutb/model-19999'
if [ ${CONV} = '1d' ];
then
RESTOREDPATH_B='C:/Data/PhD/bss_gan_palimpsest/training/DCGAN/finetune_greek_mode_full_data_ut_1d_16zb_8gut_16dut_2020-03-13-04-41/model-19900'
else
RESTOREDPATH_B="/home/as3297/projects/bss_gan_palimpsest/training/DCGAN/finetune_greek_mode_full_data_ut_background_text_antishort_16lb_8gut_16dut/model-8800"
fi
DATADIR='C:/Data/PhD/bss_gan_palimpsest/datasets/Archimedes/test_obscured'
EXPNAME="${DATANAME}_DCGAN_${CONV}"
MIXNET="mixing_net_${CONV}"
python finv_subgraphs.py -m 0.5 -bs 10 -ld ${LOGDIR} -rp_under ${RESTOREDPATH_UTB} -mnet ${MIXNET} -rp_back ${RESTOREDPATH_B} -nob 1 -dd ${DATADIR} -ims 64 -exp ${EXPNAME} -K 1 -zutb 30 -zb 16 -wb true -remix false -lrm 5.0e-3 -lru 1.0e-3 -lrb 0.5e-3 -it 50 -ot 200 -lexc 0.001 -l1 1.0 -l2 0.0
echo "Finished training"
