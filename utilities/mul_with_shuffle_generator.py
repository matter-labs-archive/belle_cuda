def gen_tables(array_len):
    even_table = {}
    odd_table = {}
    even_table_size = 0
    odd_table_size = 0
    
    #initialize dictionaries
    for i in xrange(array_len):
        even_table[i] = []
        odd_table[i] = []
    
    for a in xrange(array_len):
        for b in xrange(array_len):
            idx = int((a + b) / 2)
            if (a + b) % 2 == 0:
                even_table[idx].append((a, b))
                even_table_size = even_table_size + 1
            else:
                odd_table[idx].append((a, b))
                odd_table_size = odd_table_size + 1
                
    return (even_table, odd_table, even_table_size, odd_table_size)

from collections import namedtuple

#c = a*b + d
AsmInsn = namedtuple("AsmInsn", "op_type gen_carry use_carry dest_op first_op second_op")
op_mul = 0
op_add = 1
op_unknown = 2


def find_largest_sublist(table):
    count = 0
    index = -1
    for j in sorted(table):
        if len(table[j]) > count:
            count = len(table[j])
            index = j
    
    assert index >= 0, "Index should be greater than zero"
    return index


def to_op(index, letter):
    return letter + str(index)


def check_if_in_chain(table, table_size, index):
    if table_size == 0:
        return False
    
    new_index = find_largest_sublist(table)
    
    if (new_index == index + 1):
        return True
    elif ((new_index == index + 2) and len(table[index + 1]) > 0):
        return True
    else:
        return False


def gen_asm_for_table(table, table_size, array_len, even_flag):
    
    AsmListing = []
    main_reg = "r" if even_flag else "t"
    temp_reg = "t" if even_flag else "s"
    
    for i in xrange(array_len):
        if (len(table[i]) > 0):
            (a, b) = table[i][0]
            table[i].pop(0)
               
            insn = AsmInsn(op_mul, False, False, to_op(i, main_reg), to_op(a, "a"), to_op(b, "b"))
            AsmListing.append(insn)
            table_size = table_size - 1
           
    while(table_size > 0):
        
        index = find_largest_sublist(table)
       
        (a, b) = table[index][0]
        table[index].pop(0)
        table_size = table_size - 1
            
        insn = AsmInsn(op_mul, False, False, to_op(index, temp_reg), to_op(a, "a"), to_op(b, "b"))
        AsmListing.append(insn)
              
        start_index = index
        
        #append all muls
        
        while (check_if_in_chain(table, table_size, index)):
            index = index + 1
            (a, b) = table[index][0]
            table[index].pop(0)
            table_size = table_size - 1
            
            insn = AsmInsn(op_mul, False, False, to_op(index, temp_reg), to_op(a, "a"), to_op(b, "b"))
            AsmListing.append(insn)
            
        #append all additions
        use_carry = False
        while (start_index <= index):
            
            insn = AsmInsn(op_add, True, use_carry, to_op(start_index, main_reg), to_op(start_index, main_reg), 
                           to_op(start_index, temp_reg))
            AsmListing.append(insn)
            use_carry = True
            start_index = start_index + 1
        
        #NB: this is a small hack
        
        if (not even_flag and start_index == array_len - 1):
            insn = AsmInsn(op_add, False, True, to_op(start_index, main_reg), "0", "0")
            AsmListing.append(insn)
        else:
            insn = AsmInsn(op_add, False, True, to_op(start_index, main_reg), to_op(start_index, main_reg), "0")
            AsmListing.append(insn)
        use_carry = False
    
    return AsmListing

def gen_asm(array_len):
    even_table, odd_table, even_table_size, odd_table_size = gen_tables(array_len)
    
    AsmListing = gen_asm_for_table(even_table, even_table_size, array_len, True)
    AsmListing += gen_asm_for_table(odd_table, odd_table_size, array_len, False)
    
    return AsmListing


def generate_printable_asm(AsmListing):
    printed_asm = ""
    ending = ";\\n\\t\"\n"

    for elem in AsmListing:
        if (elem.op_type == op_mul):
            #"mul.wide.u16    s0, a1, b0;\n\t"
            
            printed_asm += "\"mul.wide.u16    " 
            printed_asm += elem.dest_op + ", " + elem.first_op + ", " + elem.second_op + ending

        elif (elem.op_type == op_add):
            printed_asm += "\"add"
            if (elem.use_carry):
                printed_asm += "c"
            printed_asm += "."
            if (elem.gen_carry):
                printed_asm += "cc."
            printed_asm += "u32    "
            
            printed_asm += elem.dest_op + ", " + elem.first_op + ", " + elem.second_op + ending

        else:
            raise ValueError('Incorrect operand type.')
     
    return printed_asm

   
ARR_LEN = 16
AsmListing = gen_asm(ARR_LEN)
print len(AsmListing)
print generate_printable_asm(AsmListing)