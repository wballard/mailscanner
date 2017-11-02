# mailscanner
Download, scan, and work on email with machine learning.

## Acquiring Email
`./bin/download-gmail` is a script that will connect over IMAP to your GMail account
downloading your messages and compiling them into a local SQLite database. From there
training and testing datasets can be concocted.

## Training Sets
The training file format is processed by `mailscanner.datasets.LabeledTextFileDataset` that uses
a relatively simple format of <label> <tab> <text> with one sample per line.

Note that this is a full in memory data-set, which aids training time, but may require sampling
if your actual source data is larger than your computer!

## Server
`mailscanner.server.server` exposes a Swagger REST service that classifies email from
text.