FROM mailscanner-base


#all the code, not including the ./var/data
COPY ./mailscanner /mailscanner/mailscanner
COPY ./bin /mailscanner/bin
COPY ./Makefile /mailscanner/
COPY ./*.txt /mailscanner/
COPY ./*.py /mailscanner/
COPY ./var/data/replies.pickle /mailscanner/var/data
COPY ./var/data/replies.weights /mailscanner/var/data

#packages needed by our server
RUN cd /mailscanner && make install


#serve up REST endpoint
USER keras
WORKDIR /mailscanner
EXPOSE 5000
CMD PORT=5000 make server