FROM mailscanner-base


# all the code
COPY ./mailscanner /mailscanner/mailscanner
COPY ./Makefile /mailscanner/

# all the trained model data
COPY ./var/data/replies.pickle /mailscanner/var/data/
COPY ./var/data/replies.weights /mailscanner/var/data/


#serve up REST endpoint
USER keras
WORKDIR /mailscanner
EXPOSE 5000
CMD PORT=5000 make server