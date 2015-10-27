import numpy as np


def makeAbun(solar_abun, abun_file, COratio, solar_times=1):
    """
    Makes the abundaces file to be used by TEA.
    The function reads Asplund et al (2009) elemental abundances file
    http://adsabs.harvard.edu/abs/2009ARA%26A..47..481A, (abudances.txt),
    sets the C/O ratio as requested, and/or multiplies the abundances of metal
    elements (all except He and H) by the desired multiplication factor. 
    The C/O ratio is set by fixing the O abundance to the solar value and
    changing the C elemental abundance so C/O ratio is as requested.

    Parameters
    ----------
    solar_abun: String
       Input Solar abundances filename.
    abun_file: String
       Output filename to store the modified elemental abundances.

    Optional parameters
    -------------------
    COratio: Float
       Desired C/O ratio.
    solar_times: Integer
       Multiplication factor for metal elemental abundances (everything
       except H and He).

    Returns
    -------
    None

    Example
    -------
    solar_abun  = 'abundances.txt'
    abun_file   = 'CO1.2abun.txt'
    COratio     = 1.2
    makeAbun(solar_abun, abun_file, COratio)

    Revisions
    ---------
    2015-10-25  Jasmina   Written by.
    """


    # Read the elemental-abundances file:
    f = open(solar_abun, 'r')
    lines = f.readlines()
    f.close()

    # Count the number of elements:
    nelem = len(lines)
    for line in lines:
        if line.startswith("#"):
            nelem -= 1

    # Allocate arrays to put information:
    index  = np.zeros(nelem, int)
    symbol = np.zeros(nelem, '|S2')
    dex    = np.zeros(nelem, np.double)
    name   = np.zeros(nelem, '|S20')
    mass   = np.zeros(nelem, np.double)

    # Store data into the arrays:
    i = 0
    for line in lines:
        if not line.startswith("#"):
            index[i], symbol[i], dex[i], name[i], mass[i] = line.strip().split()
            i += 1

    # Count the number of elements:
    nelem = len(symbol)

    # Scale the metals aundances:
    imetals = np.where((symbol != "H") & (symbol != "He"))
    dex[imetals] += np.log10(solar_times)

    # Calculate C and O abundances based on C/O requested:
    Odex = dex[np.where(symbol == "O")]
    dex[np.where(symbol == "C")] = np.log10(COratio) + Odex
    
    # Save data to file
    f = open(abun_file, "w")
    # Write header
    f.write("# Elemental abundances:\n"
            "# Columns: ordinal, symbol, dex abundances, name, molar mass.\n")
    # Write data
    for i in np.arange(nelem):
      f.write("{:3d}  {:2s}  {:5.2f}  {:10s}  {:12.8f}\n".format(
              index[i], symbol[i], dex[i], name[i], mass[i]))
    f.close()

