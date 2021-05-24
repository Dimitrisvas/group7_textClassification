import requests

HOME_URL = 'http://localhost:5000/'

COMMENT = "it's me"

def submit_comment(comment):
    return requests.post(HOME_URL, data={
        'comment_text': comment,
        'formGroupExampleInput2':''
    })

def goto_predictions():
    return requests.get(HOME_URL + 'predictions')

def goto_comments():
    return requests.get(HOME_URL + 'predictions')

def predictions_test(COMMENT):
    passed = False
    r = submit_comment(COMMENT)

    r = goto_predictions()
    if (COMMENT in r.text):
        passed = True

    return passed

def comments_test(COMMENT):
    passed = False
    COMMENTS = []

    for i in range(3):
        COMMENTS.append(f'{[i]}-{COMMENT}')
    r = submit_comment(COMMENTS[0])

    r = submit_comment(COMMENTS[1])

    r = submit_comment(COMMENTS[2])

    r = goto_comments()
    if ((COMMENTS[0] in r.text) & (COMMENTS[1] in r.text) & (COMMENTS[2] in r.text)):
        passed = True

    return passed

def error_test():
    passed = False

    r = submit_comment("")
    if ("Comment is required." in r.text):
        passed = True

    return passed

def run_tests(COMMENT):
    count = 0
    passed = []
    passed.append(predictions_test(COMMENT))
    if passed[0]:
        print("Predictions Test Passed")
        count += 1
    else:
        print("Predictions Test Failed")
    passed.append(comments_test(COMMENT))
    if passed[1]:
        print("Comments Test Passed")
        count += 1
    else:
        print("Comments Test Failed")
    passed.append(error_test())
    if passed[2]:
        print("Error Test Passed")
        count += 1
    else:
        print("Error Test Failed!")
    print (f'Passed {count}/3 tests')

run_tests(COMMENT)