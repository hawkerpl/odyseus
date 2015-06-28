import cherrypy
import optparse
import os

class HelloWorld(object):
    def __init__(self, filepath, logfile=None):
        self.filepath = filepath
        self.logfile = logfile
    @cherrypy.expose
    def index(self):
        files = os.listdir(self.filepath)
        s = ""
        for f in files:
            s += "<a href=\""+cherrypy.url(qs=cherrypy.request.query_string)+"net?filename="+str(f)+"\">"+str(f)+"</a><br>"
        return s

    @cherrypy.expose
    def net(self, filename):
        with open(os.path.join(self.filepath,filename), "r") as f:
            return f.read()

if __name__ == '__main__':
    usage = " "
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-p', type='int', default=80, help="port to host", dest='port')
    parser.add_option('--ip', type='str', default="0.0.0.0", help="ip of host", dest='ip')
    parser.add_option('-l', type='str', default=None, help="destination of log file", dest='log_filepath')
    opts, args = parser.parse_args()
    filepath = args[0]
    cherrypy.config.update({'server.socket_port': opts.port, 'server.socket_host': opts.ip})
    cherrypy.quickstart(HelloWorld(filepath,opts.log_filepath))