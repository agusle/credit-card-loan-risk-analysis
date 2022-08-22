# Third party libraries
from flask import Flask

# Local Libraries
from views import router
import settings


app = Flask(__name__)
app.secret_key = "secret key"
app.register_blueprint(router)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=settings.API_DEBUG)
