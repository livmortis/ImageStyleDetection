
class StyleException(Exception):
    error_title = "style exception: "
    code = 100
    def __init__(self, msg=''):
        self.mesg = self.error_title + msg


class ComplexException(StyleException):
    error_title = "complexity error: "
    code = 101

class ContourShapeDectectException(StyleException):
    error_title = "contour shape dectect error: "
    code = 102

class SymmetricalException(StyleException):
    error_title = "symmetrical error: "
    code = 103

class RatioException(StyleException):
    error_title = "ratio error: "
    code = 104

class ElementArrayException(StyleException):
    error_title = "element array error: "
    code = 105

class WrongParameterException(StyleException):
    error_title = "parameter error: "
    code = 106
