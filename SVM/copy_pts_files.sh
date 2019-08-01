#!/bin/bash




for sequence in {ll042t1aaunaff,ll042t1aeunaff,ll042t1afaff,ll042t1afunaff,ll042t1aiaff,ll042t1aiunaff,ll042t2afaff}; do
    for file in Images/042-ll042/$sequence/*.txt*; do
        ex -sc '1i|{' -cx "$file"
        ex -sc '1i|n_points:  66' -cx "$file"
        ex -sc '1i|version: 1' -cx "$file"
        echo "}" >> $file
    done
    rename -S _aam.txt .pts Images/042-ll042/$sequence/*_aam.txt
done

#rename -S .txt .pts *.txt
#rename -S _aam.pts .pts *_aam.pts
