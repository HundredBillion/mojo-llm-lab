"""Check one synthetic invoice response without accepting trailing generated text."""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path


def check_completion(text, business='Oak Street Bakery', total='127.50'):
    text=text.strip().removesuffix('[end of text]').strip()
    try:
        value=json.loads(text)
        if not isinstance(value,dict) or set(value)!={'Business','Total'}:
            raise ValueError('Require exactly Business and Total fields')
        if value['Business']!=business or isinstance(value['Total'],bool):
            raise ValueError('Unexpected business or amount')
        amount=Decimal(str(value['Total']))
        if not amount.is_finite() or amount!=Decimal(total):
            raise ValueError('Unexpected amount')
        return {'valid':True,'business':business,'total':str(amount)}
    except (ValueError,TypeError,InvalidOperation) as error:
        return {'valid':False,'reason':str(error)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('completion',type=Path)
    args=parser.parse_args()
    result=check_completion(args.completion.read_text())
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__=='__main__':
    main()
