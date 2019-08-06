FROM continuumio/miniconda

# Install menpo and jupyter
RUN /opt/conda/bin/conda create -n menpo python=2.7
RUN /bin/bash -c "source activate menpo && /opt/conda/bin/conda install -c menpo menpoproject -y --quiet"
RUN /opt/conda/bin/conda install jupyter -y --quiet

## Set up jupyter notebook
EXPOSE 8888
RUN mkdir /opt/notebooks
#COPY SVM_Model_v3_Adarsh.ipynb /opt/notebooks/
#COPY Images_Data.npy /opt/notebooks/
#COPY Images_Labels.np /opt/notebooks/
#COPY Image_Path.npy /opt/notebooks/

CMD ["/opt/conda/bin/jupyter", "notebook", "--notebook-dir=/opt/notebooks", "--ip='*'", "--port=8888", "--no-browser"]
