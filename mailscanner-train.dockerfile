FROM mailscanner-base


#all the code, not including the ./var/data
COPY ./mailscanner /mailscanner/mailscanner
COPY ./bin /mailscanner/bin
COPY ./Makefile /mailscanner/
COPY ./*.txt /mailscanner/
COPY ./*.py /mailscanner/

#packages needed by our server
RUN cd /mailscanner && make install

#mount in the local var 
VOLUME /mailscanner/var

#make the model
USER keras
WORKDIR /mailscanner
CMD make var/data/replies.weights